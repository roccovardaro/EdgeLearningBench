"""
RETE CONVOLUZIONALE (CNN) BASELINE
Architettura convoluzionale standard a blocchi conv configurabili (default: 32, 64, 128).
Funge da baseline di riferimento per i paradigmi avanzati ed ibridi (CNN-KAN, CNN-LTC, SNN-CNN).
"""

from typing import Sequence, Optional
import torch
import torch.nn as nn

from device import get_device
from paths import CHECKPOINT_DIR
from models.metrics import accuracy
from models.training_setup import get_training_components


# ── Singolo Blocco Convoluzionale ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Blocco Conv standard:
    Conv2d (3x3, padding=1, bias=False) → BatchNorm2d → ReLU → MaxPool2d(2)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pool_size: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


# ── Modello CNN Classifier ───────────────────────────────────────────────────

class CNNClassifier(nn.Module):
    """Rete convoluzionale baseline configurabile.

    Pipeline:
        1. Feature Extractor: sequenza di `ConvBlock` per ogni canale in `cnn_channels`
           (default: [32, 64, 128]).
        2. Global Pooling: `AdaptiveAvgPool2d((1, 1))` per garantire invarianza
           alla risoluzione spaziale in input (MNIST 28x28, CIFAR 32x32, ecc.).
        3. Flatten.
        4. Classifier Head: facoltativo stadio denso (`fc_dims`) o classificazione
           lineare diretta (`Linear(cnn_channels[-1], num_classes)`) con Dropout.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        cnn_channels: Sequence[int] = (32, 64, 128),
        fc_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.cnn_channels = tuple(cnn_channels)
        self.fc_dims = tuple(fc_dims) if fc_dims else ()
        self.dropout_rate = dropout

        # ── 1. Feature Extractor Convoluzionale ──
        layers = []
        ch_in = in_channels
        for ch_out in self.cnn_channels:
            layers.append(ConvBlock(ch_in, ch_out))
            ch_in = ch_out

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_channels = self.cnn_channels[-1] if self.cnn_channels else in_channels

        # ── 2. Classifier Head ──
        classifier_layers = []
        curr_dim = self.out_channels

        if self.fc_dims:
            for h_dim in self.fc_dims:
                if dropout > 0.0:
                    classifier_layers.append(nn.Dropout(p=dropout))
                classifier_layers.extend([
                    nn.Linear(curr_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(inplace=False),
                ])
                curr_dim = h_dim

        if dropout > 0.0:
            classifier_layers.append(nn.Dropout(p=dropout))
        classifier_layers.append(nn.Linear(curr_dim, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] → [B, num_classes]"""
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


# ── Factory ──────────────────────────────────────────────────────────────────

def build_cnn(
    in_channels: int = 3,
    num_classes: int = 10,
    cnn_channels: Sequence[int] = (32, 64, 128),
    fc_dims: Optional[Sequence[int]] = None,
    dropout: float = 0.2,
) -> CNNClassifier:
    """Costruisce un CNNClassifier configurabile e ne stampa la struttura dei parametri."""
    model = CNNClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        fc_dims=fc_dims,
        dropout=dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    conv_params = sum(p.numel() for p in model.features.parameters())
    clf_params = sum(p.numel() for p in model.classifier.parameters())
    print(f"CNN channels: {list(cnn_channels)}")
    print(f"FC dims: {list(fc_dims) if fc_dims else 'Direct Linear'}")
    print(f"Dropout: {dropout}")
    print(f"Parametri totali: {total_params:,} (Conv: {conv_params:,}, Classifier: {clf_params:,})")
    return model


# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def cnn_run(args, train_loader, test_loader, profiler=None, device=None):
    """Addestra e valuta la CNN baseline."""
    epochs = args.train.epochs
    in_channels = args.model.in_channels
    num_classes = args.model.num_classes
    cnn_channels = tuple(getattr(args.model, "cnn_channels", (32, 64, 128)))
    raw_fc_dims = getattr(args.model, "fc_dims", None)
    fc_dims = tuple(raw_fc_dims) if raw_fc_dims else ()
    dropout = getattr(args.model, "dropout", 0.2)

    device = get_device(device)

    model = build_cnn(
        in_channels=in_channels,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        fc_dims=fc_dims,
        dropout=dropout,
    ).to(device)

    optimizer, criterion = get_training_components(
        model, args, model_type="cnn"
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

        if profiler:
            profiler.log_epoch(epoch + 1, running_loss / total, correct / total)

    # ── Valutazione ─────────────────────────────────────────────
    train_acc = accuracy(model, train_loader, device)
    test_acc = accuracy(model, test_loader, device)
    print(f"Train accuracy: {train_acc:.4f}  (error: {1 - train_acc:.4f})")
    print(f"Test  accuracy: {test_acc:.4f}  (error: {1 - test_acc:.4f})")

    if profiler:
        profiler.log_eval(train_acc, test_acc)

    # ── Checkpoint ──────────────────────────────────────────────
    checkpoint_name = getattr(args.model, "checkpoint_name", "cnn.pt")
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "cnn_channels": list(cnn_channels),
            "fc_dims": list(fc_dims),
            "dropout": dropout,
        },
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model
