"""ResNet Forward-Forward.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def overlay_label_channels(images, labels, num_classes):
    """Concatena K canali one-hot; accetta etichette [B] oppure un int."""
    batch, _, height, width = images.shape
    labels = torch.as_tensor(labels, device=images.device, dtype=torch.long)
    if labels.ndim == 0:
        labels = labels.expand(batch)
    if labels.shape != (batch,):
        raise ValueError("labels deve essere uno scalare oppure avere shape [B]")
    one_hot = F.one_hot(labels, num_classes).to(dtype=images.dtype)
    one_hot = one_hot[:, :, None, None].expand(-1, -1, height, width)
    return torch.cat((images, one_hot), dim=1)


def make_negative_labels(labels, num_classes):
    """Una classe errata uniforme per campione, mai la classe corretta."""
    if num_classes < 2:
        raise ValueError("Servono almeno due classi")
    offset = torch.randint(1, num_classes, labels.shape, device=labels.device)
    return (labels + offset) % num_classes


class SampleRMSNorm(nn.Module):
    """Rimuove l'ampiezza globale mantenendo le proporzioni spaziali.

    E' una normalizzazione L2 riscalata di sqrt(C*H*W), senza parametri
    apprendibili: la media dei quadrati vale circa 1 invece di 1/(C*H*W).
    La goodness viene misurata PRIMA della normalizzazione dello stadio dopo.
    """

    def forward(self, x):
        scale = x.float().square().mean((1, 2, 3), keepdim=True)
        scale = scale.clamp_min(1e-8).sqrt().to(x.dtype)
        return x / scale


class FFResidualBlock(nn.Module):
    """BasicBlock residuo autonomo, senza dipendenze dal resto del progetto."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


def make_stage(in_channels, out_channels, count, stride=1):
    blocks = [FFResidualBlock(in_channels, out_channels, stride)]
    blocks.extend(FFResidualBlock(out_channels, out_channels) for _ in range(count - 1))
    return nn.Sequential(*blocks)


class FFConvBlock(nn.Module):
    def __init__(self, block, num_classes, threshold=2.0):
        super().__init__()
        self.block = block
        self.num_classes = num_classes
        self.threshold = float(threshold)
        self.norm = SampleRMSNorm()

    def forward(self, features, labels):
        features = self.norm(features)
        return self.block(overlay_label_channels(features, labels, self.num_classes))

    @staticmethod
    def goodness(features):
        return features.float().square().mean(dim=(1, 2, 3))



class FFResNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, layers=None, threshold=2.0):
        super().__init__()
        layers = list(layers) if layers is not None else [2, 2, 2]
        if len(layers) != 3 or any(not isinstance(n, int) or n < 1 for n in layers):
            raise ValueError("layers deve contenere tre interi positivi")
        if num_classes < 2 or in_channels < 1 or threshold <= 0:
            raise ValueError("Configurazione non valida")
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = layers
        self.threshold = float(threshold)
        self.history = []

        # I dieci canali di etichetta vengono aggiunti da FFConvBlock.
        first_stage = nn.Sequential(
            nn.Conv2d(in_channels + num_classes, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            make_stage(16, 16, layers[0]),
        )
        self.blocks = nn.ModuleList([
            FFConvBlock(first_stage, num_classes, threshold),
            FFConvBlock(make_stage(16 + num_classes, 32, layers[1], 2), num_classes, threshold),
            FFConvBlock(make_stage(32 + num_classes, 64, layers[2], 2), num_classes, threshold),
        ])
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _candidate_goodness(self, images, n_blocks):
        """[B, classi, blocchi]; il chiamante gestisce eval/no_grad."""
        scores = []
        for candidate in range(self.num_classes):
            features = images
            local = []
            for block in self.blocks[:n_blocks]:
                features = block(features, candidate)
                local.append(block.goodness(features))
            scores.append(torch.stack(local, dim=1))
        return torch.stack(scores, dim=1)

    @torch.no_grad()
    def score(self, images, label):
        """Goodness cumulativa escludendo il primo layer (label-dominated)."""
        self.eval()
        features = images
        total = torch.zeros(images.shape[0], device=images.device)
        for idx, block in enumerate(self.blocks):
            features = block(features, label)
            if idx == 0:
                continue  # Il blocco 1 riflette la separazione triviale della label
            total += block.goodness(features)
        self.train()
        return total

    @torch.no_grad()
    def predict(self, images):
        self.eval()
        scores = self._candidate_goodness(images, len(self.blocks))
        self.train()
        # Esclude il primo layer (indice 0) dalla somma della goodness
        return scores[:, :, 1:].sum(dim=2).argmax(dim=1)



    def fit(self, train_loader, epochs_per_layer=100, lr=0.001, device="cuda"):
        """Addestramento greedy; anche una seconda chiamata riabilita i gradienti."""
        if epochs_per_layer < 1 or lr <= 0:
            raise ValueError("epochs_per_layer e lr devono essere positivi")
        self.to(device)
        self.eval()
        self.requires_grad_(False)
        self.history = []

        for block_idx, block in enumerate(self.blocks):
            print(f"\nTraining block {block_idx + 1}/{len(self.blocks)}")
            block.requires_grad_(True)
            block.train()
            optimizer = Adam(block.parameters(), lr=lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs_per_layer)

            for epoch in range(epochs_per_layer):
                total_loss = total_pos = total_neg = total_pairs = 0.0
                total_samples = 0
                progress = tqdm(train_loader, desc=f"Block {block_idx + 1} Epoch {epoch + 1}")
                for images, labels in progress:
                    images = images.to(device)
                    labels = labels.to(device=device, dtype=torch.long)
                    bs = labels.shape[0]

                    neg_labels = make_negative_labels(labels, self.num_classes)
                    paired_labels = torch.cat((labels, neg_labels), dim=0)
                    features = torch.cat((images, images), dim=0)
                    with torch.no_grad():
                        for prev in self.blocks[:block_idx]:
                            # Sono congelati e in eval: running stats invarianti.
                            features = prev(features, paired_labels)

                    optimizer.zero_grad(set_to_none=True)
                    # Un unico forward: BatchNorm comune a positivi e negativi.
                    h = block(features, paired_labels)
                    g_pos, g_neg = block.goodness(h).split(bs, dim=0)
                    loss = 0.5 * (
                        F.softplus(block.threshold - g_pos).mean()
                        + F.softplus(g_neg - block.threshold).mean()
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Loss non finita nel blocco {block_idx + 1}")
                    loss.backward()
                    nn.utils.clip_grad_norm_(block.parameters(), max_norm=5.0, error_if_nonfinite=True)
                    optimizer.step()

                    total_loss += loss.item() * bs
                    total_pos += g_pos.detach().sum().item()
                    total_neg += g_neg.detach().sum().item()
                    total_pairs += (g_pos.detach() > g_neg.detach()).sum().item()
                    total_samples += bs
                    if hasattr(progress, "set_postfix"):
                        progress.set_postfix(loss=f"{loss.item():.4f}")

                if total_samples == 0:
                    raise ValueError("train_loader e' vuoto")
                record = {
                    "block": block_idx + 1,
                    "epoch": epoch + 1,
                    "loss": total_loss / total_samples,
                    "g_pos": total_pos / total_samples,
                    "g_neg": total_neg / total_samples,
                    "pair_accuracy": total_pairs / total_samples,
                }
                self.history.append(record)
                print(
                    f"  loss={record['loss']:.4f} | g+={record['g_pos']:.4f} | "
                    f"g-={record['g_neg']:.4f} | pair_acc={record['pair_accuracy']:.2%}"
                )
                scheduler.step()

            block.eval()
            block.requires_grad_(False)
            block.zero_grad(set_to_none=True)

        self.eval()
        return self


@torch.no_grad()
def accuracy(model, loader, device="cuda"):
    correct = total = 0
    model.eval()
    for images, labels in tqdm(loader, desc="Evaluation"):
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.long)
        correct += (model.predict(images) == labels).sum().item()
        total += labels.shape[0]
    model.train()
    if total == 0:
        raise ValueError("Il loader di valutazione e' vuoto")
    return correct / total


def ff_resnet_run(args, train_loader, test_loader):
    # Mantiene il percorso di salvataggio del progetto originale.
    from paths import CHECKPOINT_DIR

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model_config = {
        "num_classes": args.model.num_classes,
        "in_channels": args.model.in_channels,
        "threshold": args.model.threshold,
        "layers": [2, 2, 2],
    }
    print(f"Device: {device}")
    model = FFResNet(**model_config)
    model.fit(train_loader, args.train.epochs_per_layer, args.train.lr, device)
    # Se train_loader applica augmentation, la train accuracy le include.
    train_acc = accuracy(model, train_loader, device)
    test_acc = accuracy(model, test_loader, device)
    print(f"Train accuracy: {train_acc:.2%}")
    print(f"Test accuracy:  {test_acc:.2%}")

    checkpoint_path = Path(CHECKPOINT_DIR) / args.model.checkpoint_name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "history": model.history,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")
    return model

