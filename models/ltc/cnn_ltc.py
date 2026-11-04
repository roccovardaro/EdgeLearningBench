"""
CNN-LTC: modello ibrido con CNN Feature Extractor + Liquid Time-Constant Network (senza AutoNCP).

Combina un front-end convoluzionale (invarianza spaziale 2D e weight sharing)
con una cella LTC ricorrente a tempo continuo (dinamiche ODE per l'integrazione temporale).
Questo approccio elimina l'esplosione dei parametri causata dalle proiezioni dense da pixel grezzi.
"""
import torch
import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import FullyConnected

from device import get_device
from paths import CHECKPOINT_DIR
from models.metrics import accuracy
from models.training_setup import get_training_components


# ── CNN Feature Extractor ────────────────────────────────────────────────────

class CNNFeatureExtractor(nn.Module):
    """CNN a blocchi conv per estrarre feature visive 2D da immagini.

    Architettura per blocco:
        Conv2d (3x3, padding=1) → BatchNorm2d → GELU → MaxPool2d(2)
    """

    def __init__(self, in_channels=3, cnn_channels=(32, 64, 128)):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out in cnn_channels:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            ch_in = ch_out

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = cnn_channels[-1]

    def forward(self, x):
        """[B, C, H, W] → [B, out_dim]"""
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


# ── Modello CNN-LTC ──────────────────────────────────────────────────────────

class CNNLiquidNetwork(nn.Module):
    """Rete ibrida CNN + Liquid Time-Constant (senza AutoNCP).

    Pipeline:
        1. CNN Feature Extractor: estrae feature spaziali [B, C, H, W] → [B, cnn_out_dim]
        2. Proiezione temporale: proietta e suddivide il feature vector compatto
           in `num_steps` token sequenziali → [B, num_steps, proj_dim]
        3. Elaborazione LTC: la sequenza viene processata dalla cella LTC
           con connettività FullyConnected standard (senza AutoNCP)
        4. Classificazione: l'output dell'ultimo timestep viene proiettato
           sulle classi di output

    Attributi:
        num_steps : numero di timestep nella sequenza
        units     : numero di neuroni nella cella LTC
        proj_dim  : dimensione di ogni token dopo la proiezione (default = units)
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        cnn_channels=(32, 64, 128),
        units=64,
        num_steps=8,
        dropout=0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.units = units
        self.num_steps = num_steps

        # ── 1. CNN Feature Extractor ──
        self.cnn = CNNFeatureExtractor(in_channels=in_channels, cnn_channels=cnn_channels)
        self.cnn_out_dim = self.cnn.out_dim

        # ── 2. Proiezione Feature compatto → sequenza di token ──
        self.proj_dim = units
        self.proj = nn.Sequential(
            nn.Linear(self.cnn_out_dim, self.proj_dim * num_steps),
            nn.LayerNorm(self.proj_dim * num_steps),
            nn.GELU(),
        )

        # ── 3. Cella LTC (Fully-Connected standard, senza AutoNCP) ──
        wiring = FullyConnected(units=units)
        self.ltc = LTC(
            input_size=self.proj_dim,
            units=wiring,
            return_sequences=True,
        )

        # ── 4. Layer Norm + Head di classificazione ──
        self.classifier = nn.Sequential(
            nn.LayerNorm(units),
            nn.Dropout(p=dropout),
            nn.Linear(units, num_classes),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Tensor [B, C, H, W].

        Returns:
            logits: [B, num_classes]
        """
        B = x.size(0)

        # 1. Estrazione feature visive
        feat = self.cnn(x)  # [B, cnn_out_dim]

        # 2. Proiezione in sequenza temporale
        tokens = self.proj(feat)  # [B, proj_dim * num_steps]
        tokens = tokens.view(B, self.num_steps, self.proj_dim)  # [B, T, proj_dim]

        # 3. Elaborazione LTC
        ltc_out, _ = self.ltc(tokens)  # [B, T, units]

        # 4. Output dell'ultimo timestep e classificazione
        last_out = ltc_out[:, -1, :]  # [B, units]
        logits = self.classifier(last_out)

        return logits


# ── Factory ──────────────────────────────────────────────────────────────────

def build_cnn_ltc(
    in_channels=3,
    num_classes=10,
    cnn_channels=(32, 64, 128),
    units=64,
    num_steps=8,
    dropout=0.2,
):
    """Costruisce una CNNLiquidNetwork con i parametri specificati."""
    model = CNNLiquidNetwork(
        in_channels=in_channels,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        units=units,
        num_steps=num_steps,
        dropout=dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    cnn_params = sum(p.numel() for p in model.cnn.parameters())
    proj_params = sum(p.numel() for p in model.proj.parameters())
    ltc_params = sum(p.numel() for p in model.ltc.parameters())
    cls_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"CNN channels: {list(cnn_channels)} (in={in_channels} → out={model.cnn_out_dim})")
    print(f"LTC units: {units} | Timesteps: {num_steps} | Dropout: {dropout}")
    print(f"Wiring: FullyConnected (units={units}) [senza AutoNCP]")
    print(
        f"Parametri — CNN: {cnn_params:,} | Proj: {proj_params:,} | "
        f"LTC: {ltc_params:,} | Head: {cls_params:,} | Totale: {total_params:,}"
    )
    return model




# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def cnn_ltc_run(args, train_loader, test_loader, profiler=None, device=None):
    """Addestra e valuta una CNN-LTC (Convolutional Liquid Time-Constant Network)."""
    epochs = args.train.epochs
    weight_decay = getattr(args.train, "weight_decay", 1e-4)

    in_channels = args.model.in_channels
    num_classes = args.model.num_classes
    cnn_channels = tuple(args.model.cnn_channels)
    units = getattr(args.model, "units", 64)
    num_steps = getattr(args.model, "num_steps", 8)
    dropout = getattr(args.model, "dropout", 0.2)

    device = get_device(device)

    model = build_cnn_ltc(
        in_channels=in_channels,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        units=units,
        num_steps=num_steps,
        dropout=dropout,
    ).to(device)

    optimizer, criterion = get_training_components(
        model, args, model_type="cnn_ltc"
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

        if profiler:
            profiler.log_epoch(epoch + 1, epoch_loss, train_acc)

        if budget_stop:
            break

    # ── Valutazione ─────────────────────────────────────────────
    train_acc = accuracy(model, train_loader, device)
    test_acc = accuracy(model, test_loader, device)
    print(f"Train accuracy: {train_acc:.4f}  (error: {1 - train_acc:.4f})")
    print(f"Test  accuracy: {test_acc:.4f}  (error: {1 - test_acc:.4f})")

    if profiler:
        profiler.log_eval(train_acc, test_acc)

    # ── Checkpoint ──────────────────────────────────────────────
    checkpoint_path = CHECKPOINT_DIR / args.model.checkpoint_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "cnn_channels": list(cnn_channels),
            "units": units,
            "num_steps": num_steps,
            "dropout": dropout,
        },
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model
