"""
LTC – Liquid Time-Constant Network per classificazione di immagini.

Utilizza la libreria `ncps` (Neural Circuit Policies) con cella LTC
senza AutoNCP (connettività Fully-Connected standard).

Paper: "Liquid Time-constant Networks" (Hasani et al., AAAI 2021)
Repo:  https://github.com/mlech26l/ncps

L'input immagine viene convertito in una sequenza temporale (proiezione
lineare + reshape in token) e processato step-by-step dalla rete LTC.
"""
import torch
import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import FullyConnected

from device import get_device
from paths import CHECKPOINT_DIR
from models.metrics import accuracy
from models.training_setup import get_training_components


# ── Modello ──────────────────────────────────────────────────────────────────

class LiquidNetwork(nn.Module):
    """Rete Liquid Time-Constant per classificazione di immagini.

    Pipeline:
        1. Proiezione dell'input: l'immagine appiattita [B, D] viene proiettata
           e suddivisa in `num_steps` token sequenziali → [B, num_steps, proj_dim]
        2. Elaborazione temporale: la sequenza viene processata dalla cella LTC
           con connettività FullyConnected (tutti i neuroni interconnessi, senza AutoNCP)
        3. Classificazione: l'output dell'ultimo timestep viene proiettato
           sulle classi di output

    Attributi:
        num_steps   : numero di timestep nella sequenza
        units       : numero di neuroni nella cella LTC
        proj_dim    : dimensione di ogni token dopo la proiezione
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        units=64,
        num_steps=16,
        dropout=0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.units = units

        # ── Proiezione input → sequenza di token ──
        # Dimensione di ogni token: usiamo `units` come proj_dim
        self.proj_dim = units
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.proj_dim * num_steps),
            nn.LayerNorm(self.proj_dim * num_steps),
            nn.GELU(),
        )

        # ── Wiring Fully-Connected (senza AutoNCP) ──
        # Connettività standard all-to-all tra i neuroni della cella LTC,
        wiring = FullyConnected(units=units)

        # ── Cella LTC ──
        # La cella LTC di ncps accetta input [B, T, input_size] e restituisce
        # output [B, T, units].
        self.ltc = LTC(
            input_size=self.proj_dim,
            units=wiring,
            return_sequences=True,
        )

        # ── Layer Norm sull'output + head di classificazione ──
        self.classifier = nn.Sequential(
            nn.LayerNorm(units),
            nn.Dropout(p=dropout),
            nn.Linear(units, num_classes),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Tensor [B, C, H, W] o [B, D].

        Returns:
            logits: [B, num_classes]
        """
        B = x.size(0)
        x = x.flatten(1)  # [B, D]

        # Proiezione e reshape in sequenza
        tokens = self.input_proj(x)                          # [B, proj_dim * num_steps]
        tokens = tokens.view(B, self.num_steps, self.proj_dim)  # [B, T, proj_dim]

        # Elaborazione LTC
        ltc_out, _ = self.ltc(tokens)   # ltc_out: [B, T, units]

        # Prendiamo l'output dell'ultimo timestep
        last_out = ltc_out[:, -1, :]    # [B, units]

        # Classificazione
        logits = self.classifier(last_out)

        return logits


# ── Factory ──────────────────────────────────────────────────────────────────

def build_ltc(
    input_dim,
    num_classes,
    units=64,
    num_steps=16,
    dropout=0.2,
):
    """Costruisce una LiquidNetwork con i parametri specificati."""
    model = LiquidNetwork(
        input_dim=input_dim,
        num_classes=num_classes,
        units=units,
        num_steps=num_steps,
        dropout=dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LTC config: input_dim={input_dim} → units={units} → classes={num_classes}")
    print(f"Timesteps: {num_steps} | Dropout: {dropout}")
    print(f"Wiring: FullyConnected (units={units}) [senza AutoNCP]")
    print(f"Parametri totali: {total_params:,}")
    return model



# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def ltc_run(args, train_loader, test_loader, profiler=None, device=None):
    """Addestra e valuta una Liquid Time-Constant Network (ncps)."""
    epochs = args.train.epochs
    weight_decay = getattr(args.train, "weight_decay", 1e-4)

    input_dim = args.model.input_dim
    num_classes = args.model.num_classes
    units = getattr(args.model, "units", 64)
    num_steps = getattr(args.model, "num_steps", 16)
    dropout = getattr(args.model, "dropout", 0.2)

    device = get_device(device)

    model = build_ltc(
        input_dim=input_dim,
        num_classes=num_classes,
        units=units,
        num_steps=num_steps,
        dropout=dropout,
    ).to(device)

    optimizer, criterion = get_training_components(
        model, args, model_type="ltc"
    )

    budget_tracker = getattr(args, "budget_tracker", getattr(profiler, "budget_tracker", None))

    # ── Training ────────────────────────────────────────────────
    budget_stop = False
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if budget_tracker and budget_tracker.static_fwd_macs == 0:
                budget_tracker.attach_model(model, sample_input=images[:1], device=device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            # Gradient clipping per stabilità ODE
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (logits.detach().argmax(1) == labels).sum().item()
            total += labels.size(0)

            if budget_tracker and budget_tracker.step(batch_size=labels.size(0), epoch=epoch + 1):
                budget_stop = True
                break

        epoch_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"loss={epoch_loss:.4f} | "
            f"train_acc={train_acc:.4f}"
        )
        profiler.log_epoch(epoch + 1, epoch_loss, train_acc)

        if budget_stop:
            break

    # ── Valutazione ─────────────────────────────────────────────
    args.edge_profiler.stop()

    train_acc = accuracy(model, train_loader, device)
    test_acc = accuracy(model, test_loader, device)
    print(f"Train accuracy: {train_acc:.4f}  (error: {1 - train_acc:.4f})")
    print(f"Test  accuracy: {test_acc:.4f}  (error: {1 - test_acc:.4f})")
    profiler.log_eval(train_acc, test_acc)

    # ── Checkpoint ──────────────────────────────────────────────
    checkpoint_path = CHECKPOINT_DIR / args.model.checkpoint_name
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "units": units,
            "num_steps": num_steps,
            "dropout": dropout,
        },
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model
