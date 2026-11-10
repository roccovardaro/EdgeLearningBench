"""
SNN-ResNet (Spiking Residual Network) con SpikingJelly e multi-step mode.

Adatta l'architettura ResNet classica (BasicBlock e ResNet) a Spiking Neural Network,
utilizzando layer SpikingJelly multi-step (step_mode='m'), neuroni ParametricLIFNode,
surrogate gradient descent (ATan) e rate coding / elaborazione temporale vettorizzata.
"""
from typing import Any, List, Optional
from pathlib import Path
import torch
import torch.nn as nn

from spikingjelly.activation_based import layer, neuron, surrogate, functional

from device import get_device
from paths import CHECKPOINT_DIR, RESULTS_DIR
from profiler.snn_profiler import SNNProfiler
from models.metrics import accuracy
from models.training_setup import get_training_components


# ── Blocco Residuo Spiking ───────────────────────────────────────────────────

class SpikingBasicBlock(nn.Module):
    """BasicBlock spiking equivalente al BasicBlock di ResNet.

    Conv2d → BatchNorm2d → LIFNode → Conv2d → BatchNorm2d (+ identity) → LIFNode
    """
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        tau: float = 20.0,
        surrogate_function=None,
    ):
        super().__init__()
        if surrogate_function is None:
            surrogate_function = surrogate.ATan()

        self.conv1 = layer.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, step_mode="m"
        )
        self.bn1 = layer.BatchNorm2d(planes, step_mode="m")
        self.lif1 = neuron.ParametricLIFNode(
            init_tau=tau,
            decay_input=False,
            surrogate_function=surrogate_function,
            step_mode="m",
        )

        self.conv2 = layer.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"
        )
        self.bn2 = layer.BatchNorm2d(planes, step_mode="m")
        self.lif2 = neuron.ParametricLIFNode(
            init_tau=tau,
            decay_input=False,
            surrogate_function=surrogate_function,
            step_mode="m",
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagazione multi-step [T, B, C, H, W]."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.lif2(out)

        return out


# ── Modello SpikingResNet ─────────────────────────────────────────────────────

class SpikingResNet(nn.Module):
    """ResNet spiking multi-step (step_mode='m') con neuroni ParametricLIFNode.

    Architettura:
        Input rate-coded [T, B, C, H, W]
        → Conv1 (16) → BN1 → LIF1
        → Layer1 (16 ch) → Layer2 (32 ch) → Layer3 (64 ch)
        → AdaptiveAvgPool2d(1) → Flatten → Linear → LIFNode (output)
    """

    def __init__(
        self,
        block,
        layers: List[int],
        num_classes: int,
        in_channels: int = 3,
        beta: float = 0.95,
        num_steps: int = 16,
        encoding: str = "rate",
    ):
        super().__init__()
        self.inplanes = 16
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

        # Convoluzione iniziale spiking
        self.conv1 = layer.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"
        )
        self.bn1 = layer.BatchNorm2d(16, step_mode="m")
        self.lif1 = neuron.ParametricLIFNode(
            init_tau=tau,
            decay_input=False,
            surrogate_function=spike_grad,
            step_mode="m",
        )

        self.layer1 = self._make_layer(
            block, 16, layers[0], stride=1, tau=tau, surrogate_function=spike_grad
        )
        self.layer2 = self._make_layer(
            block, 32, layers[1], stride=2, tau=tau, surrogate_function=spike_grad
        )
        self.layer3 = self._make_layer(
            block, 64, layers[2], stride=2, tau=tau, surrogate_function=spike_grad
        )

        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode="m")
        self.flatten = layer.Flatten(step_mode="m")
        self.fc = layer.Linear(64 * block.expansion, num_classes, step_mode="m")
        self.output_lif = neuron.ParametricLIFNode(
            init_tau=tau,
            decay_input=False,
            surrogate_function=spike_grad,
            step_mode="m",
            store_v_seq=True,
        )

        # Inizializzazione pesi analoga a resnet.py
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, layer.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, layer.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        tau: float = 20.0,
        surrogate_function=None,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    step_mode="m",
                ),
                layer.BatchNorm2d(planes * block.expansion, step_mode="m"),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                tau=tau,
                surrogate_function=surrogate_function,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    tau=tau,
                    surrogate_function=surrogate_function,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Forward pass multi-step con rate coding.

        Args:
            x: Tensor [B, C, H, W]

        Returns:
            spk_out: Spike di output [num_steps, B, num_classes]
            mem_out: Potenziale di membrana registrato [num_steps, B, num_classes]
        """
        # Reset delle membrane ad ogni passaggio
        functional.reset_net(self)

        # Ripetizione temporale multi-step [num_steps, B, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.num_steps, *([1] * x.ndim))
        if self.encoding == "rate":
            rate = x_seq.clamp(0.0, 1.0)
            cur = torch.bernoulli(rate)
        else:
            cur = x_seq

        # Convoluzione iniziale spiking
        cur = self.conv1(cur)
        cur = self.bn1(cur)
        cur = self.lif1(cur)

        # Stadi residui
        cur = self.layer1(cur)
        cur = self.layer2(cur)
        cur = self.layer3(cur)

        # Global pooling, flatten e classificazione FC spiking
        cur = self.avgpool(cur)
        cur = self.flatten(cur)
        cur = self.fc(cur)
        spk_out = self.output_lif(cur)
        mem_out = self.output_lif.v_seq

        return spk_out, mem_out


# ── Factory ──────────────────────────────────────────────────────────────────

def _snn_resnet(
    arch: str,
    layers: List[int],
    num_classes: int,
    in_channels: int = 3,
    beta: float = 0.95,
    num_steps: int = 16,
    encoding: str = "rate",
    weights_path: Optional[str] = None,
    **kwargs: Any,
) -> SpikingResNet:
    model = SpikingResNet(
        block=SpikingBasicBlock,
        layers=layers,
        num_classes=num_classes,
        in_channels=in_channels,
        beta=beta,
        num_steps=num_steps,
        encoding=encoding,
    )

    if weights_path is not None:
        ckpt = CHECKPOINT_DIR / weights_path
        if not ckpt.exists():
            ckpt = Path(weights_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {weights_path}")

        try:
            state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        except Exception:
            state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)

        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        initial_state = model.state_dict()
        if "conv1.weight" in state_dict and state_dict["conv1.weight"].shape != model.conv1.weight.shape:
            state_dict["conv1.weight"] = initial_state["conv1.weight"]

        if "fc.weight" in state_dict and state_dict["fc.weight"].shape != model.fc.weight.shape:
            state_dict["fc.weight"] = initial_state["fc.weight"]
            if "fc.bias" in state_dict and model.fc.bias is not None:
                state_dict["fc.bias"] = initial_state["fc.bias"]

        # Caricamento con strict=False per tollerare parametri LIF aggiunti o pesi non-spiking
        model.load_state_dict(state_dict, strict=False)
        print(f"[{arch}] Pesi caricati da {ckpt}")

    return model


def snn_resnet(
    num_classes: int,
    in_channels: int = 3,
    layers: Optional[List[int]] = None,
    beta: float = 0.95,
    num_steps: int = 16,
    encoding: str = "rate",
    weights_path: Optional[str] = None,
    **kwargs: Any,
) -> SpikingResNet:
    """Costruttore standard per SNN-ResNet."""
    if layers is None:
        layers = [3, 3, 3]
    depth = sum(layers) * 2 + 2
    return _snn_resnet(
        arch=f"SNN-ResNet-{depth}",
        layers=layers,
        num_classes=num_classes,
        in_channels=in_channels,
        beta=beta,
        num_steps=num_steps,
        encoding=encoding,
        weights_path=weights_path,
        **kwargs,
    )


def build_snn_resnet(
    num_classes: int,
    in_channels: int = 3,
    layers: Optional[List[int]] = None,
    beta: float = 0.95,
    num_steps: int = 16,
    encoding: str = "rate",
    weights_path: Optional[str] = None,
) -> SpikingResNet:
    """Costruisce un SpikingResNet e stampa le statistiche dell'architettura."""
    if layers is None:
        layers = [3, 3, 3]
    depth = sum(layers) * 2 + 2
    model = snn_resnet(
        num_classes=num_classes,
        in_channels=in_channels,
        layers=layers,
        beta=beta,
        num_steps=num_steps,
        encoding=encoding,
        weights_path=weights_path,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SNN-ResNet layers: {layers} (SNN-ResNet-{depth}) | in_channels={in_channels} | classes={num_classes}")
    print(f"Timesteps: {num_steps} | Beta: {beta} (Tau: {model.tau:.2f}) | Encoding: {model.encoding}")
    print(f"Parametri totali: {total_params:,}")
    return model


# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def snn_resnet_run(args, train_loader, test_loader, profiler=None, device=None):
    """Addestra e valuta una SNN-ResNet con SpikingJelly."""
    epochs = args.train.epochs
    in_channels = args.model.in_channels
    num_classes = args.model.num_classes
    layers = list(getattr(args.model, "layers", [3, 3, 3]))
    beta = getattr(args.model, "beta", 0.95)
    num_steps = getattr(args.model, "num_steps", 16)
    encoding = getattr(args.model, "encoding", "rate")
    weights_path = getattr(args.model, "weights_path", None)

    device = get_device(device)
    if device.type == "mps":
        print("[SpikingJelly] I neuroni SpikingJelly supportano solo CPU e NVIDIA CUDA (non Apple Silicon MPS).")
        print("[SpikingJelly] L'esecuzione viene commutata automaticamente su CPU.")
        device = torch.device("cpu")

    model = build_snn_resnet(
        num_classes=num_classes,
        in_channels=in_channels,
        layers=layers,
        beta=beta,
        num_steps=num_steps,
        encoding=encoding,
        weights_path=weights_path,
    ).to(device)

    optimizer, criterion = get_training_components(
        model, args, model_type="snn_resnet"
    )
    snn_prof = SNNProfiler(model=model, tag="snn_resnet", output_dir=RESULTS_DIR)
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
            snn_prof.ensure_static_macs(images[:1])

            optimizer.zero_grad()
            spk_out, mem_out = model(images)
            loss = criterion(spk_out, mem_out, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Reset canonico SpikingJelly dopo l'aggiornamento dei pesi
            functional.reset_net(model)

            running_loss += loss.item() * labels.size(0)

            # Valutazione accuratezza durante l'epoca (basata su potenziale medio di membrana)
            mem_avg = mem_out.mean(dim=0).detach()
            correct += (mem_avg.argmax(1) == labels).sum().item()
            total += labels.size(0)

            snn_prof.update(spk_out)

            fr = snn_prof.current_firing_rate if getattr(snn_prof, "current_firing_rate", 0.0) > 0 else (spk_out.sum().item() / spk_out.numel() if spk_out.numel() > 0 else 0.0)
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
        snn_prof.log_epoch(epoch + 1)

        if budget_stop:
            break

    # ── Valutazione ─────────────────────────────────────────────
    train_acc = accuracy(model, train_loader, device)

    snn_prof.reset()
    test_acc = accuracy(model, test_loader, device)
    snn_prof.log_epoch("test", tag="test")

    print(f"Train accuracy: {train_acc:.4f}  (error: {1 - train_acc:.4f})")
    print(f"Test  accuracy: {test_acc:.4f}  (error: {1 - test_acc:.4f})")

    if profiler:
        profiler.log_eval(train_acc, test_acc)

    snn_prof.print_summary()
    snn_prof.export_csv()

    # ── Checkpoint ──────────────────────────────────────────────
    checkpoint_path = CHECKPOINT_DIR / args.model.checkpoint_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "layers": layers,
            "beta": beta,
            "tau": model.tau,
            "num_steps": num_steps,
            "encoding": model.encoding,
            "backend": "spikingjelly",
        },
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model
