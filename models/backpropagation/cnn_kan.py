"""CNN-KAN: modello ibrido con CNN feature extractor + KAN classifier.

Una piccola CNN (3 blocchi conv) estrae feature spaziali e riduce la
dimensionalità dell'immagine.  I layer KAN (B-spline) lavorano poi su
uno spazio compatto, risultando più efficienti e accurati rispetto
al KAN puro applicato direttamente sui pixel.

"""

import torch
import torch.nn as nn
from efficient_kan import KAN

from paths import CHECKPOINT_DIR


# ── CNN Feature Extractor ────────────────────────────────────────────────────

class CNNFeatureExtractor(nn.Module):
    """Piccola CNN a 3 blocchi conv per estrarre feature spaziali.

    Architettura per blocco:
        Conv2d → BatchNorm2d → ReLU → MaxPool2d(2)

    Dopo i 3 blocchi, un AdaptiveAvgPool2d(1) + Flatten produce
    un vettore di dimensione uguale ai canali dell'ultimo blocco.
    """

    def __init__(self, in_channels=3, cnn_channels=(32, 64, 128)):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out in cnn_channels:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
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


# ── Modello ibrido ───────────────────────────────────────────────────────────

class CNNKANClassifier(nn.Module):
    """CNN come feature extractor + KAN come classificatore.

    La CNN riduce l'immagine a un vettore compatto (es. 128 dim),
    poi il KAN applica B-spline su questo spazio ridotto.
    """

    def __init__(self, in_channels=3, num_classes=10,
                 cnn_channels=(32, 64, 128),
                 hidden_dims=(64,),
                 grid_size=5, spline_order=3):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_channels, cnn_channels)

        # Dimensioni dei layer KAN: [cnn_out] + hidden + [num_classes]
        kan_input_dim = self.cnn.out_dim
        layers_hidden = [kan_input_dim] + list(hidden_dims) + [num_classes]
        self.kan = KAN(
            layers_hidden=layers_hidden,
            grid_size=grid_size,
            spline_order=spline_order,
        )

        self.grid_size = grid_size
        self.spline_order = spline_order

    def forward(self, x):
        """[B, C, H, W] → logits [B, num_classes]"""
        features = self.cnn(x)
        return self.kan(features)

    def regularization_loss(self):
        """Regularizzazione L1 sugli spline weights."""
        return self.kan.regularization_loss()


# ── Factory ──────────────────────────────────────────────────────────────────

def build_cnn_kan(in_channels=3, num_classes=10,
                  cnn_channels=(32, 64, 128),
                  hidden_dims=(64,),
                  grid_size=5, spline_order=3):
    """Costruisce un CNNKANClassifier con i parametri specificati."""
    model = CNNKANClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        hidden_dims=hidden_dims,
        grid_size=grid_size,
        spline_order=spline_order,
    )
    total_params = sum(p.numel() for p in model.parameters())
    cnn_params = sum(p.numel() for p in model.cnn.parameters())
    kan_params = sum(p.numel() for p in model.kan.parameters())
    print(f"CNN channels: {list(cnn_channels)}")
    print(f"CNN output dim: {model.cnn.out_dim}")
    print(f"KAN layers: {[model.cnn.out_dim] + list(hidden_dims) + [num_classes]}")
    print(f"Grid size: {grid_size}, Spline order: {spline_order}")
    print(f"Parametri — CNN: {cnn_params:,}  KAN: {kan_params:,}  Totale: {total_params:,}")
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

def cnn_kan_run(args, train_loader, test_loader):
    """Addestra e valuta un CNN-KAN con backpropagation standard."""
    # Parametri di training
    epochs = args.train.epochs
    lr = args.train.lr
    weight_decay = getattr(args.train, "weight_decay", 1e-4)
    reg_lambda = getattr(args.train, "regularization_lambda", 0.01)

    # Parametri del modello
    in_channels = args.model.in_channels
    num_classes = args.model.num_classes
    cnn_channels = tuple(args.model.cnn_channels)
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
    model = build_cnn_kan(
        in_channels=in_channels,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
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
            "in_channels": in_channels,
            "num_classes": num_classes,
            "cnn_channels": list(cnn_channels),
            "hidden_dims": list(hidden_dims),
            "grid_size": grid_size,
            "spline_order": spline_order,
        },
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model
