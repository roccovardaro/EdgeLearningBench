"""
SNN-CNN (Spiking Convolutional Neural Network) con SpikingJelly e multi-step mode.

Architettura che combina blocchi convoluzionali con neuroni
ParametricLIFNode di SpikingJelly in multi-step mode (step_mode='m').
L'output è costituito dagli spike e dal potenziale di membrana
cumulati sull'ultimo layer LIF.
"""
from pathlib import Path
import torch
import torch.nn as nn

from spikingjelly.activation_based import layer, neuron, surrogate, functional

from device import get_device
from paths import CHECKPOINT_DIR, RESULTS_DIR
from profiler.snn_profiler import SNNProfiler
from models.metrics import accuracy
from models.training_setup import get_training_components


# ── Blocco Conv Spiking ──────────────────────────────────────────────────────

class SpikingConvBlock(nn.Module):
    """Singolo blocco convoluzionale spiking con SpikingJelly in multi-step mode.

    Conv2d → BatchNorm2d → LIFNode → MaxPool2d (tutti con step_mode='m')
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
        self.conv = layer.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
            step_mode="m",
        )
        self.bn = layer.BatchNorm2d(out_channels, step_mode="m")
        self.lif = neuron.ParametricLIFNode(
            init_tau=tau,
            decay_input=False,
            surrogate_function=surrogate_function,
            step_mode="m",
        )
        self.pool = layer.MaxPool2d(
            kernel_size=pool_size, stride=pool_size, step_mode="m"
        )

    def forward(self, x):
        """Propagazione multi-step [T, B, C, H, W].

        Args:
            x: Tensor [num_steps, B, C_in, H, W]

        Returns:
            spk: Spike in uscita [num_steps, B, C_out, H', W']
        """
        x = self.conv(x)
        x = self.bn(x)
        spk = self.lif(x)
        spk = self.pool(spk)
        return spk


# ── Modello ──────────────────────────────────────────────────────────────────

class SpikingCNN(nn.Module):
    """SNN-CNN con blocchi convoluzionali spiking e classificatore FC spiking.

    Architettura multi-step (step_mode='m'):
        Rate coding / espansione temporale [T, B, C, H, W]
        → [SpikingConvBlock × N] → AdaptiveAvgPool2d(1) → Flatten → Dropout
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
        encoding="rate",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.beta = beta
        encoding = str(encoding).strip().lower()
        if encoding not in ("direct", "rate"):
            raise ValueError(f"Encoding non valido: {encoding!r}. Usa 'direct' oppure 'rate'.")
        self.encoding = encoding

        # Relazione tau e decadimento beta: tau = 1 / (1 - beta)
        tau = 1.0 / (1.0 - beta) if (beta is not None and beta < 1.0) else 20.0
        self.tau = tau

        spike_grad = surrogate.ATan()
        self.spike_grad = spike_grad

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

        # ── Pooling globale, flatten e dropout multi-step ──
        self.global_pool = layer.AdaptiveAvgPool2d(1, step_mode="m")
        self.flatten = layer.Flatten(step_mode="m")
        self.dropout = layer.Dropout(p=dropout, step_mode="m")

        # ── Classificatore FC spiking multi-step ──
        self.fc = layer.Linear(cnn_channels[-1], num_classes, step_mode="m")
        self.output_lif = neuron.ParametricLIFNode(
            init_tau=tau,
            decay_input=False,
            surrogate_function=spike_grad,
            step_mode="m",
            store_v_seq=True,
        )

    def forward(self, x):
        """Forward pass multi-step con rate coding.

        Args:
            x: Tensor [B, C, H, W]

        Returns:
            spk_out: Spike di output accumulati [num_steps, B, num_classes]
            mem_out: Potenziale di membrana accumulato [num_steps, B, num_classes]
        """
        # Reset delle membrane all'inizio di ogni inferenza/passaggio
        functional.reset_net(self)

        # Ripetizione temporale multi-step [num_steps, B, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.num_steps, *([1] * x.ndim))
        if self.encoding == "rate":
            rate = x_seq.clamp(0.0, 1.0)
            cur = torch.bernoulli(rate)
        else: #direct encoding
            cur = x_seq

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
        mem_out = self.output_lif.v_seq

        return spk_out, mem_out


# ── Factory ──────────────────────────────────────────────────────────────────

def build_snn_cnn(
    in_channels=3,
    num_classes=10,
    cnn_channels=(32, 64, 128),
    beta=0.95,
    num_steps=25,
    dropout=0.2,
    encoding="rate",
):
    """Costruisce un SpikingCNN con i parametri specificati."""
    model = SpikingCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        beta=beta,
        num_steps=num_steps,
        dropout=dropout,
        encoding=encoding,
    )
    total_params = sum(p.numel() for p in model.parameters())
    conv_params = sum(
        p.numel() for block in model.conv_blocks for p in block.parameters()
    )
    fc_params = sum(p.numel() for p in model.fc.parameters())
    print(f"SNN-CNN channels: {list(cnn_channels)}")
    print(f"Timesteps: {num_steps} | Beta: {beta} (Tau: {model.tau:.2f}) | Dropout: {dropout} | Encoding: {model.encoding}")
    print(f"Parametri — Conv: {conv_params:,}  FC: {fc_params:,}  Totale: {total_params:,}")
    return model


# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def snn_cnn_run(args, train_loader, test_loader, profiler=None, device=None):
    """Addestra e valuta un SNN-CNN con SpikingJelly in multi-step mode."""
    epochs = args.train.epochs
    weight_decay = getattr(args.train, "weight_decay", 1e-4)

    in_channels = args.model.in_channels
    num_classes = args.model.num_classes
    cnn_channels = tuple(args.model.cnn_channels)
    beta = getattr(args.model, "beta", 0.95)
    num_steps = getattr(args.model, "num_steps", 25)
    dropout = getattr(args.model, "dropout", 0.2)
    encoding = getattr(args.model, "encoding", "rate")

    device = get_device(device)
    if device.type == "mps":
        print("[SpikingJelly] I neuroni SpikingJelly supportano solo CPU e NVIDIA CUDA (non Apple Silicon MPS).")
        print("[SpikingJelly] L'esecuzione viene commutata automaticamente su CPU.")
        device = torch.device("cpu")

    model = build_snn_cnn(
        in_channels=in_channels,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        beta=beta,
        num_steps=num_steps,
        dropout=dropout,
        encoding=encoding,
    ).to(device)

    optimizer, criterion = get_training_components(
        model, args, model_type="snn_cnn"
    )
    snn_profiler = SNNProfiler(model=model, tag="snn_cnn", output_dir=RESULTS_DIR)
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
            snn_profiler.ensure_static_macs(images[:1])

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

            fr = snn_profiler.current_firing_rate if getattr(snn_profiler, "current_firing_rate", 0.0) > 0 else (spk_out.sum().item() / spk_out.numel() if spk_out.numel() > 0 else 0.0)
            if budget_tracker and budget_tracker.step(
                batch_size=labels.size(0),
                firing_rate=fr,
                num_steps_snn=num_steps,
                epoch=epoch + 1,
            ):
                budget_stop = True
                break

        epoch_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"loss={epoch_loss:.4f} | "
            f"train_acc={train_acc:.4f}"
        )

        if profiler:
            profiler.log_epoch(epoch + 1, epoch_loss, train_acc)
        snn_profiler.log_epoch(epoch + 1)

        if budget_stop:
            break

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
            "encoding": model.encoding,
            "backend": "spikingjelly",
        },
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model
