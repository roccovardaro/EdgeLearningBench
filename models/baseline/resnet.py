import torch
import torch.nn as nn
from typing import Any, List, Optional

from device import get_device
from paths import CHECKPOINT_DIR
from models.metrics import accuracy
from models.training_setup import get_training_components

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, in_channels=3):
        super(ResNet, self).__init__()
        self.inplanes = 16

        #CONVOLUZIONE INZIALE
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)

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
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers) #nn.Sequential(blocco1, blocco2, blocco3)

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

def resnet(
    num_classes: int,
    in_channels: int = 3,
    layers: Optional[List[int]] = None,
    weights_path: Optional[str] = None,
    **kwargs: Any
) -> ResNet:
    if layers is None:
        layers = [9, 9, 9]
    depth = sum(layers) * 2 + 2
    return _resnet(
        arch=f"ResNet-{depth}",
        layers=layers,
        num_classes=num_classes,
        in_channels=in_channels,
        weights_path=weights_path,
        **kwargs,
    )


# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def resnet_run(args, train_loader, test_loader, profiler=None, device=None):
    """Addestra e valuta una ResNet."""
    epochs = args.train.epochs
    in_channels = args.model.in_channels
    num_classes = args.model.num_classes
    layers = list(getattr(args.model, "layers", [9, 9, 9]))
    depth = sum(layers) * 2 + 2
    model_name = f"ResNet-{depth}"

    device = get_device(device)

    model = resnet(
        num_classes=num_classes,
        in_channels=in_channels,
        layers=layers,
        weights_path=getattr(args.model, "weights_path", None),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet layers: {layers} ({model_name}) | in_channels={in_channels} | classes={num_classes}")
    print(f"Parametri totali: {total_params:,}")

    optimizer, criterion = get_training_components(
        model, args, model_type="resnet"
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
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
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
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "layers": layers,
        },
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model