"""
Utilizza l'implementazione efficient-kan di Blealtan
(https://github.com/Blealtan/efficient-kan) per costruire una rete KAN
basata su B-spline, addestrata con backpropagation standard.
"""

import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from efficient_kan import KAN

from paths import CHECKPOINT_DIR


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
        # LayerNorm normalizza i pixel grezzi nell'intervallo ≈ [-1, 1],
        # che è il grid_range di default delle B-spline di efficient-kan.
        self.input_norm = nn.LayerNorm(input_dim)

        # Proiezione lineare: riduce 3072 → bottleneck_dim in modo
        # economico (parametri lineari), così i layer KAN lavorano
        # su uno spazio compatto.
        self.projection = nn.Linear(input_dim, bottleneck_dim)

        # ── KAN layers ──────────────────────────────────────────
        layers_hidden = [bottleneck_dim] + self.hidden_dims + [num_classes]
        self.kan = KAN(
            layers_hidden=layers_hidden,
            grid_size=grid_size,
            spline_order=spline_order,
        )

        # ── Dropout tra i layer KAN ─────────────────────────────
        # Viene applicato sull'output della proiezione e tra i layer
        # interni del KAN (vedi forward).
        self.dropout = nn.Dropout(p=dropout)

        # Inizializzazione Kaiming per la proiezione lineare
        nn.init.kaiming_normal_(self.projection.weight, a=math.sqrt(5))
        nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        """Forward pass con flatten automatico [B, C, H, W] → [B, D]."""
        x = x.flatten(1)
        x = self.input_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return self.kan(x)

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
    norm_params = sum(p.numel() for p in model.input_norm.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    print(f"KAN layers: {[bottleneck_dim] + list(hidden_dims) + [num_classes]}")
    print(f"Grid size: {grid_size}, Spline order: {spline_order}")
    print(f"Bottleneck: {input_dim} → {bottleneck_dim}, Dropout: {dropout}")
    print(f"Parametri — Norm: {norm_params:,}  Proj: {proj_params:,}  "
          f"KAN: {kan_params:,}  Totale: {total_params:,}")
    return model


# ── Valutazione ──────────────────────────────────────────────────────────────

@torch.no_grad()
def accuracy(model, loader, device):
    """Calcola l'accuratezza su un DataLoader."""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def kan_run(args, train_loader, test_loader):
    """Addestra e valuta un KAN con backpropagation standard."""
    # Parametri di training
    epochs = args.train.epochs
    lr = args.train.lr
    weight_decay = getattr(args.train, "weight_decay", 1e-4)
    reg_lambda = getattr(args.train, "regularization_lambda", 0.01)

    # Parametri del modello
    input_dim = args.model.input_dim
    num_classes = args.model.num_classes
    hidden_dims = tuple(args.model.hidden_dims)
    grid_size = getattr(args.model, "grid_size", 5)
    spline_order = getattr(args.model, "spline_order", 3)
    bottleneck_dim = getattr(args.model, "bottleneck_dim", 256)
    dropout = getattr(args.model, "dropout", 0.2)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ── Scheduler: warmup lineare + cosine annealing ────────────
    warmup_epochs = min(5, epochs // 4) if epochs > 4 else 0
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Label smoothing migliora la generalizzazione evitando che il modello diventi troppo "confident" sulle predizioni.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

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
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

    # ── Valutazione ─────────────────────────────────────────────
    train_acc = accuracy(model, train_loader, device)
    test_acc = accuracy(model, test_loader, device)
    print(f"Train accuracy: {train_acc:.4f}  (error: {1 - train_acc:.4f})")
    print(f"Test  accuracy: {test_acc:.4f}  (error: {1 - test_acc:.4f})")

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

