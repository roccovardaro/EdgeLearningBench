"""
MLP (Multi-Layer Perceptron) per classificazione di immagini, addestrato con backpropagation standard.
"""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from paths import CHECKPOINT_DIR


# ── Modello ──────────────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """MLP con flatten automatico dell'input e layer nascosti configurabili.

    Architettura per blocco: Linear → BatchNorm1d → ReLU → Dropout
    """

    def __init__(self, input_dim, num_classes, hidden_dims=(512, 256),
                 dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass con flatten automatico [B, C, H, W] → [B, D]."""
        x = x.flatten(1)
        return self.net(x)


# ── Factory ──────────────────────────────────────────────────────────────────

def build_mlp(input_dim, num_classes, hidden_dims=(512, 256), dropout=0.2):
    """Costruisce un MLPClassifier con i parametri specificati."""
    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MLP layers: {[input_dim] + list(hidden_dims) + [num_classes]}")
    print(f"Dropout: {dropout}")
    print(f"Parametri totali: {total_params:,}")
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

def mlp_run(args, train_loader, test_loader):
    """Addestra e valuta un MLP con backpropagation standard."""
    epochs = args.train.epochs
    lr = args.train.lr
    weight_decay = getattr(args.train, "weight_decay", 1e-4)

    input_dim = args.model.input_dim
    num_classes = args.model.num_classes
    hidden_dims = tuple(args.model.hidden_dims)
    dropout = getattr(args.model, "dropout", 0.2)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    model = build_mlp(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
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
            loss.backward()
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
            "dropout": dropout,
        },
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model
