"""
SNN-CNN (Spiking Convolutional Neural Network) con rate coding.

Architettura ibrida che combina blocchi convoluzionali con neuroni
Leaky Integrate-and-Fire (LIF) di snntorch. L'input è codificato
in spike tramite rate coding e l'output è il potenziale di membrana
cumulato sull'ultimo layer LIF.
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from device import get_device
from paths import CHECKPOINT_DIR, RESULTS_DIR
from profiler.snn_profiler import SNNProfiler
from models.metrics import accuracy
from models.training_setup import get_training_components


# ── Blocco Conv Spiking ──────────────────────────────────────────────────────

class SpikingConvBlock(nn.Module):
    """Singolo blocco convoluzionale spiking.

    Conv2d → BatchNorm2d → LIF → MaxPool2d
    """

    def __init__(self, in_channels, out_channels, beta, spike_grad,
                 kernel_size=3, pool_size=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x, mem):
        """Propagazione di un singolo timestep.

        Args:
            x:   Tensor [B, C_in, H, W]
            mem: Stato della membrana del LIF

        Returns:
            spk: Spike in uscita [B, C_out, H', W']
            mem: Membrana aggiornata
        """
        x = self.conv(x)
        x = self.bn(x)
        spk, mem = self.lif(x, mem)
        spk = self.pool(spk)
        return spk, mem


# ── Modello ──────────────────────────────────────────────────────────────────

class SpikingCNN(nn.Module):
    """SNN-CNN con blocchi convoluzionali spiking e classificatore FC spiking.

    Architettura:
        [SpikingConvBlock × N] → AdaptiveAvgPool2d(1) → Flatten
        → Linear → LIF (output)

    L'input viene presentato come spike rate-coded per `num_steps` timestep.
    L'output è la membrana cumulata sull'ultimo layer LIF.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        cnn_channels=(32, 64, 128),
        beta=0.95,
        num_steps=25,
        dropout=0.2,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # ── Blocchi convoluzionali spiking ──
        self.conv_blocks = nn.ModuleList()
        ch_in = in_channels
        for ch_out in cnn_channels:
            self.conv_blocks.append(
                SpikingConvBlock(ch_in, ch_out, beta=beta, spike_grad=spike_grad)
            )
            ch_in = ch_out

        # ── Pooling globale ──
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(p=dropout)

        # ── Classificatore FC spiking ──
        self.fc = nn.Linear(cnn_channels[-1], num_classes)
        self.output_lif = snn.Leaky(
            beta=beta, spike_grad=spike_grad, learn_beta=True, output=True,
        )

    def forward(self, x):
        """Forward pass con rate coding.

        Args:
            x: Tensor [B, C, H, W]

        Returns:
            spk_out: Spike di output accumulati [num_steps, B, num_classes]
            mem_out: Potenziale di membrana accumulato [num_steps, B, num_classes]
        """
        batch_size = x.shape[0]

        # ── Inizializza le membrane di tutti i LIF ──
        conv_mems = [block.lif.init_leaky() for block in self.conv_blocks]
        mem_output = self.output_lif.init_leaky()

        # ── Registri per le uscite ──
        spk_recordings = []
        mem_recordings = []

        # ── Simulazione temporale con rate coding ──
        for step in range(self.num_steps):
            
            #TODO:implementare rate coding
            # Rate coding: spike con probabilità proporzionale all'intensità
            #rate = x.clamp(0.0, 1.0)
            #inp = torch.bernoulli(rate)

            #Direct coding
            inp = x

            # Propagazione attraverso i blocchi conv spiking
            cur = inp
            for i, block in enumerate(self.conv_blocks):
                cur, conv_mems[i] = block(cur, conv_mems[i])

            # Pooling globale + flatten + dropout
            cur = self.global_pool(cur)
            cur = self.flatten(cur)
            cur = self.dropout(cur)

            # Classificatore FC spiking
            cur = self.fc(cur)
            spk_out, mem_output = self.output_lif(cur, mem_output)

            spk_recordings.append(spk_out)
            mem_recordings.append(mem_output)

        return torch.stack(spk_recordings), torch.stack(mem_recordings)


# ── Factory ──────────────────────────────────────────────────────────────────

def build_snn_cnn(
    in_channels=3,
    num_classes=10,
    cnn_channels=(32, 64, 128),
    beta=0.95,
    num_steps=25,
    dropout=0.2,
):
    """Costruisce un SpikingCNN con i parametri specificati."""
    model = SpikingCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        beta=beta,
        num_steps=num_steps,
        dropout=dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    conv_params = sum(
        p.numel() for block in model.conv_blocks for p in block.parameters()
    )
    fc_params = sum(p.numel() for p in model.fc.parameters())
    print(f"SNN-CNN channels: {list(cnn_channels)}")
    print(f"Timesteps: {num_steps} | Beta: {beta} | Dropout: {dropout}")
    print(f"Parametri — Conv: {conv_params:,}  FC: {fc_params:,}  Totale: {total_params:,}")
    return model


# SNNCNNLoss è importata da training_setup



# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def snn_cnn_run(args, train_loader, test_loader, profiler=None, device=None):
    """Addestra e valuta un SNN-CNN con rate coding e backpropagation surrogata."""
    epochs = args.train.epochs
    lr = args.train.lr
    weight_decay = getattr(args.train, "weight_decay", 1e-4)

    in_channels = args.model.in_channels
    num_classes = args.model.num_classes
    cnn_channels = tuple(args.model.cnn_channels)
    beta = getattr(args.model, "beta", 0.95)
    num_steps = getattr(args.model, "num_steps", 25)
    dropout = getattr(args.model, "dropout", 0.2)

    mem_weight = getattr(args.train, "mem_weight", 1.0)
    spike_weight = getattr(args.train, "spike_weight", 1.0)

    device = get_device(device)

    model = build_snn_cnn(
        in_channels=in_channels,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        beta=beta,
        num_steps=num_steps,
        dropout=dropout,
    ).to(device)

    optimizer, scheduler, criterion = get_training_components(
        model, args, model_type="snn_cnn"
    )
    snn_profiler = SNNProfiler(tag="snn_cnn", output_dir=RESULTS_DIR)

    # ── Training ────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            spk_out, mem_out = model(images)
            loss = criterion(spk_out, mem_out, labels)
            loss.backward()
            # Gradient clipping per stabilità con surrogate gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            mem_avg = mem_out.mean(dim=0).detach()
            correct += (mem_avg.argmax(1) == labels).sum().item()
            total += labels.size(0)
            snn_profiler.update(spk_out)

        scheduler.step()

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"loss={running_loss / total:.4f} | "
            f"train_acc={correct / total:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if profiler:
            profiler.log_epoch(epoch + 1, running_loss / total, correct / total, scheduler.get_last_lr()[0])
        snn_profiler.log_epoch(epoch + 1)

    # ── Valutazione ─────────────────────────────────────────────
    train_acc = accuracy(model, train_loader, device)
    test_acc = accuracy(model, test_loader, device)
    print(f"Train accuracy: {train_acc:.4f}  (error: {1 - train_acc:.4f})")
    print(f"Test  accuracy: {test_acc:.4f}  (error: {1 - test_acc:.4f})")

    if profiler:
        profiler.log_eval(train_acc, test_acc)

    snn_profiler.print_summary()
    snn_profiler.export_csv()

    # ── Checkpoint ──────────────────────────────────────────────
    checkpoint_path = CHECKPOINT_DIR / args.model.checkpoint_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "cnn_channels": list(cnn_channels),
            "beta": beta,
            "num_steps": num_steps,
            "dropout": dropout,
        },
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model
