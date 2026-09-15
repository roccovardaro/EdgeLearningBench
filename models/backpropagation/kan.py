"""KAN: Kolmogorov-Arnold Network con addestramento via backpropagation.

Utilizza l'implementazione efficient-kan di Blealtan
(https://github.com/Blealtan/efficient-kan) per costruire una rete KAN
basata su B-spline, addestrata con backpropagation standard.

Installazione dipendenza:
    pip install git+https://github.com/Blealtan/efficient-kan.git
"""

import torch
import torch.nn as nn
from efficient_kan import KAN

from paths import CHECKPOINT_DIR


# ── Modello ──────────────────────────────────────────────────────────────────

class KANClassifier(nn.Module):
    """Wrapper attorno a efficient-kan.KAN per classificazione di immagini.

    Gestisce automaticamente il flatten dell'input e costruisce
    la rete KAN con le dimensioni specificate.
    """

    def __init__(self, input_dim, num_classes, hidden_dims=(64, 64),
                 grid_size=5, spline_order=3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = list(hidden_dims)
        self.grid_size = grid_size
        self.spline_order = spline_order

        layers_hidden = [input_dim] + self.hidden_dims + [num_classes]
        self.kan = KAN(
            layers_hidden=layers_hidden,
            grid_size=grid_size,
            spline_order=spline_order,
        )

    def forward(self, x):
        """Forward pass con flatten automatico [B, C, H, W] → [B, D]."""
        x = x.flatten(1)
        return self.kan(x)

    def regularization_loss(self):
        """Proxy per la regularizzazione L1 sugli spline weights di efficient-kan."""
        return self.kan.regularization_loss()


# ── Factory ──────────────────────────────────────────────────────────────────

def build_kan(input_dim, num_classes, hidden_dims=(64, 64),
              grid_size=5, spline_order=3):
    """Costruisce un KANClassifier con i parametri specificati."""
    model = KANClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        grid_size=grid_size,
        spline_order=spline_order,
    )
    print(f"KAN layers: {[input_dim] + list(hidden_dims) + [num_classes]}")
    print(f"Grid size: {grid_size}, Spline order: {spline_order}")
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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

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
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"loss={running_loss / total:.4f} | "
            f"train_acc={correct / total:.4f}"
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
        },
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model
