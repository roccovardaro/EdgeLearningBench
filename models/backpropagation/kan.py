"""
Utilizza l'implementazione efficient-kan di Blealtan
(https://github.com/Blealtan/efficient-kan) per costruire una rete KAN
basata su B-spline, addestrata con backpropagation standard.
"""

import math
import torch
import torch.nn as nn
from efficient_kan import KAN

from device import get_device
from paths import CHECKPOINT_DIR
from models.metrics import accuracy
from models.training_setup import get_training_components


# ── Modello ──────────────────────────────────────────────────────────────────

class KANClassifier(nn.Module):
    """Wrapper attorno a efficient-kan.KAN per classificazione di immagini.
    """

    def __init__(self, input_dim, num_classes, hidden_dims=(64, 64),
                 grid_size=5, spline_order=3, bottleneck_dim=256,
                 dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = list(hidden_dims)
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.bottleneck_dim = bottleneck_dim

        # ── Pre-processing ──────────────────────────────────────
        # LayerNorm normalizza i pixel grezzi
        self.input_norm = nn.LayerNorm(input_dim)

        # Proiezione lineare: riduce input_dim → bottleneck_dim in modo
        # economico (parametri lineari), così i layer KAN lavorano
        # su uno spazio compatto.
        self.projection = nn.Linear(input_dim, bottleneck_dim)

        # LayerNorm sul bottleneck: assicura che le attivazioni in ingresso
        # al primo layer KAN siano centrate e scalate nell'intervallo ≈ [-1, 1],
        # che è il grid_range ideale delle B-spline di efficient-kan.
        self.proj_norm = nn.LayerNorm(bottleneck_dim)

        # ── KAN layers ──────────────────────────────────────────
        layers_hidden = [bottleneck_dim] + self.hidden_dims + [num_classes]
        self.kan = KAN(
            layers_hidden=layers_hidden,
            grid_size=grid_size,
            spline_order=spline_order,
        )

        # ── Normalizzazione e Dropout tra i layer KAN ───────────
        # LayerNorm previene la deriva di scala tra layer KAN consecutivi,
        # mantenendo le attivazioni nel dominio di supporto delle spline.
        self.hidden_norms = nn.ModuleList([nn.LayerNorm(h) for h in self.hidden_dims])
        self.dropout = nn.Dropout(p=dropout)

        # Inizializzazione Kaiming per la proiezione lineare
        nn.init.kaiming_normal_(self.projection.weight, a=math.sqrt(5))
        nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        """Forward pass con flatten automatico [B, C, H, W] → [B, D]."""

        x = x.flatten(1)
        x = self.input_norm(x)
        x = self.projection(x)
        x = self.proj_norm(x)
        x = self.dropout(x)

        # Passaggio sequenziale attraverso i layer KAN:
        # Per i soli layer nascosti (escluso l'output finale) applica:
        # 1. Skip connection (se le dimensioni coincidono) per facilitare i gradienti.
        # 2. LayerNorm per mantenere i valori nel range [-1, 1] attivo delle B-spline.
        # 3. Dropout per ridurre l'overfitting.

        for i, layer in enumerate(self.kan.layers):
            res = x
            x = layer(x)
            # Applica residual (se dimensioni compatibili), norm e dropout
            # solo sui layer intermedi (non sull'output logits finale)
            if i < len(self.kan.layers) - 1:
                if x.shape == res.shape:
                    x = x + res
                x = self.hidden_norms[i](x)
                x = self.dropout(x)

        return x

    def regularization_loss(self):
        """Proxy per la regularizzazione L1 sugli spline weights di efficient-kan."""
        return self.kan.regularization_loss()


# ── Factory ──────────────────────────────────────────────────────────────────

def build_kan(input_dim, num_classes, hidden_dims=(64, 64),
              grid_size=5, spline_order=3, bottleneck_dim=256,
              dropout=0.2):
    """Costruisce un KANClassifier con i parametri specificati."""
    model = KANClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        grid_size=grid_size,
        spline_order=spline_order,
        bottleneck_dim=bottleneck_dim,
        dropout=dropout,
    )

    # Stampa il breakdown dei parametri
    proj_params = sum(p.numel() for p in model.projection.parameters())
    kan_params = sum(p.numel() for p in model.kan.parameters())
    norm_params = (
        sum(p.numel() for p in model.input_norm.parameters())
        + sum(p.numel() for p in model.proj_norm.parameters())
        + sum(p.numel() for p in model.hidden_norms.parameters())
    )
    total_params = sum(p.numel() for p in model.parameters())

    print(f"KAN layers: {[bottleneck_dim] + list(hidden_dims) + [num_classes]}")
    print(f"Grid size: {grid_size}, Spline order: {spline_order}")
    print(f"Bottleneck: {input_dim} → {bottleneck_dim}, Dropout: {dropout}")
    print(f"Parametri — Norm: {norm_params:,}  Proj: {proj_params:,}  "
          f"KAN: {kan_params:,}  Totale: {total_params:,}")
    return model



# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def kan_run(args, train_loader, test_loader, profiler=None, device=None):
    """Addestra e valuta un KAN con backpropagation standard."""
    # Parametri di training
    epochs = args.train.epochs
    lr = args.train.lr
    weight_decay = getattr(args.train, "weight_decay", 1e-4)
    reg_lambda = getattr(args.train, "regularization_lambda", 0.0)

    # Parametri del modello
    input_dim = args.model.input_dim
    num_classes = args.model.num_classes
    hidden_dims = tuple(args.model.hidden_dims)
    grid_size = getattr(args.model, "grid_size", 5)
    spline_order = getattr(args.model, "spline_order", 3)
    bottleneck_dim = getattr(args.model, "bottleneck_dim", 256)
    dropout = getattr(args.model, "dropout", 0.2)

    device = get_device(device)

    # Costruzione e spostamento su device
    model = build_kan(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        grid_size=grid_size,
        spline_order=spline_order,
        bottleneck_dim=bottleneck_dim,
        dropout=dropout,
    ).to(device)

    optimizer, scheduler, criterion = get_training_components(
        model, args, model_type="kan"
    )

    # ── Training ────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Regularizzazione L1 sugli spline weights (da efficient-kan)
            if reg_lambda > 0:
                loss = loss + reg_lambda * model.regularization_loss()

            loss.backward()

            # Gradient clipping: le B-spline possono generare gradienti
            # molto grandi, specialmente nelle prime epoche.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"loss={running_loss / total:.4f} | "
            f"train_acc={correct / total:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if profiler:
            profiler.log_epoch(epoch + 1, running_loss / total, correct / total, scheduler.get_last_lr()[0])

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
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dims": list(hidden_dims),
            "grid_size": grid_size,
            "spline_order": spline_order,
            "bottleneck_dim": bottleneck_dim,
            "dropout": dropout,
        },
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model

