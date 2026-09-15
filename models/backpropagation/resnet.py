import torch
import torch.nn as nn
from typing import Any, List, Optional

from paths import CHECKPOINT_DIR

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, in_channels=3):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(in_channels, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(
    arch: str,
    layers: List[int],
    weights_path: Optional[str] = None,
    **kwargs: Any
) -> ResNet:
    model = ResNet(BasicBlock, layers, **kwargs)

    if weights_path is not None:
        ckpt = CHECKPOINT_DIR / weights_path
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint non trovato: {ckpt}"
            )
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)

        initial_state = model.state_dict()

        # Se cambia il numero di canali, mantiene
        # l'inizializzazione della prima convoluzione.
        if state_dict["conv1.weight"].shape != model.conv1.weight.shape:
            state_dict["conv1.weight"] = initial_state["conv1.weight"]

        # Se cambia il numero di classi, mantiene
        # l'inizializzazione del classificatore.
        if state_dict["fc.weight"].shape != model.fc.weight.shape:
            state_dict["fc.weight"] = initial_state["fc.weight"]
            state_dict["fc.bias"] = initial_state["fc.bias"]

        model.load_state_dict(state_dict)
        print(f"[{arch}] Pesi caricati da {ckpt}")

    return model

def resnet56(
    num_classes,
    in_channels=3,
    weights_path=None,
):
    return _resnet(
        arch="resnet56",
        layers=[9, 9, 9],
        num_classes=num_classes,
        in_channels=in_channels,
        weights_path=weights_path,
    )


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

def resnet_run(args, train_loader, test_loader):
    """Addestra e valuta una ResNet con backpropagation standard."""
    epochs = args.train.epochs
    lr = args.train.lr
    in_channels = args.model.in_channels
    num_classes = args.model.num_classes

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    model = resnet56(num_classes=num_classes, in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
        },
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model