"""
SNN-CNN (Spiking Convolutional Neural Network) con SpikingJelly.

Architettura che combina blocchi convoluzionali con neuroni
Leaky Integrate-and-Fire (LIFNode) di SpikingJelly.
L'output è costituito dagli spike e dal potenziale di membrana
cumulati sull'ultimo layer LIF.
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional
from device import get_device
from paths import CHECKPOINT_DIR, RESULTS_DIR
from profiler.snn_profiler import SNNProfiler
from models.metrics import accuracy
from models.training_setup import get_training_components


# ── Blocco Conv Spiking ──────────────────────────────────────────────────────

class SpikingConvBlock(nn.Module):
    """Singolo blocco convoluzionale spiking con SpikingJelly.

    Conv2d → BatchNorm2d → LIFNode → MaxPool2d
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        tau,
        surrogate_function,
        kernel_size=3,
        pool_size=2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = neuron.ParametricLIFNode(
            init_tau=tau,
            decay_input=False,
            surrogate_function=surrogate_function,
            step_mode="s",
        )
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        """Propagazione di un singolo timestep.

        Args:
            x: Tensor [B, C_in, H, W]

        Returns:
            spk: Spike in uscita [B, C_out, H', W']
        """
        x = self.conv(x)
        x = self.bn(x)
        spk = self.lif(x)
        spk = self.pool(spk)
        return spk


# ── Modello ──────────────────────────────────────────────────────────────────

class SpikingCNN(nn.Module):
    """SNN-CNN con blocchi convoluzionali spiking e classificatore FC spiking.

    Architettura:
        [SpikingConvBlock × N] → AdaptiveAvgPool2d(1) → Flatten
        → Linear → LIFNode (output)

    L'output fornisce spike e potenziali di membrana registrati per num_steps timestep.
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

        # Relazione tau e decadimento beta: tau = 1 / (1 - beta)
        tau = 1.0 / (1.0 - beta) if (beta is not None and beta < 1.0) else 20.0
        self.tau = tau

        spike_grad = surrogate.ATan()

        # ── Blocchi convoluzionali spiking ──
        self.conv_blocks = nn.ModuleList()
        ch_in = in_channels
        for ch_out in cnn_channels:
            self.conv_blocks.append(
                SpikingConvBlock(
                    ch_in, ch_out, tau=tau, surrogate_function=spike_grad
                )
            )
            ch_in = ch_out

        # ── Pooling globale ──
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(p=dropout)

        # ── Classificatore FC spiking ──
        self.fc = nn.Linear(cnn_channels[-1], num_classes)
        self.output_lif = neuron.ParametricLIFNode(
            init_tau=tau,
            decay_input=False,
            surrogate_function=spike_grad,
            step_mode="s",
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Tensor [B, C, H, W]

        Returns:
            spk_out: Spike di output accumulati [num_steps, B, num_classes]
            mem_out: Potenziale di membrana accumulato [num_steps, B, num_classes]
        """
        # Reset delle membrane all'inizio di ogni inferenza/passaggio
        functional.reset_net(self)

        spk_recordings = []
        mem_recordings = []

        for _ in range(self.num_steps):
            cur = x

            # Propagazione attraverso i blocchi conv spiking
            for block in self.conv_blocks:
                cur = block(cur)

            # Pooling globale + flatten + dropout
            cur = self.global_pool(cur)
            cur = self.flatten(cur)
            cur = self.dropout(cur)

            # Classificatore FC spiking
            cur = self.fc(cur)
            spk_out = self.output_lif(cur)

            spk_recordings.append(spk_out)
            mem_recordings.append(self.output_lif.v)

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
    print(f"Timesteps: {num_steps} | Beta: {beta} (Tau: {model.tau:.2f}) | Dropout: {dropout}")
    print(f"Parametri — Conv: {conv_params:,}  FC: {fc_params:,}  Totale: {total_params:,}")
    return model


# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def snn_cnn_run(args, train_loader, test_loader, profiler=None, device=None):
    """Addestra e valuta un SNN-CNN con SpikingJelly."""
    epochs = args.train.epochs
    weight_decay = getattr(args.train, "weight_decay", 1e-4)

    in_channels = args.model.in_channels
    num_classes = args.model.num_classes
    cnn_channels = tuple(args.model.cnn_channels)
    beta = getattr(args.model, "beta", 0.95)
    num_steps = getattr(args.model, "num_steps", 25)
    dropout = getattr(args.model, "dropout", 0.2)

    device = get_device(device)

    model = build_snn_cnn(
        in_channels=in_channels,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        beta=beta,
        num_steps=num_steps,
        dropout=dropout,
    ).to(device)

    optimizer, criterion = get_training_components(
        model, args, model_type="snn_cnn"
    )
    snn_profiler = SNNProfiler(model=model, tag="snn_cnn", output_dir=RESULTS_DIR)

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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Reset canonico SpikingJelly dopo l'aggiornamento dei pesi
            functional.reset_net(model)

            running_loss += loss.item() * labels.size(0)

            # Valutazione accuratezza
            mem_avg = mem_out.mean(dim=0).detach()
            correct += (mem_avg.argmax(1) == labels).sum().item()
            total += labels.size(0)
            snn_profiler.update(spk_out)

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"loss={running_loss / total:.4f} | "
            f"train_acc={correct / total:.4f}"
        )

        if profiler:
            profiler.log_epoch(epoch + 1, running_loss / total, correct / total)
        snn_profiler.log_epoch(epoch + 1)

    # ── Valutazione ─────────────────────────────────────────────
    train_acc = accuracy(model, train_loader, device)

    snn_profiler.reset()
    test_acc = accuracy(model, test_loader, device)
    snn_profiler.log_epoch("test", tag="test")

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
            "tau": model.tau,
            "num_steps": num_steps,
            "dropout": dropout,
            "backend": "spikingjelly",
        },
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model
