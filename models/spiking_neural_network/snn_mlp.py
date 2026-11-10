"""
SNN-MLP (Spiking Neural Network – Multi-Layer Perceptron) con rate coding e SpikingJelly.

Utilizza neuroni Leaky Integrate-and-Fire (LIFNode) di SpikingJelly, con rate coding
per la codifica degli input e surrogate gradient descent per l'addestramento.
"""
import torch
import torch.nn as nn

from spikingjelly.activation_based import layer, neuron, surrogate, functional

from device import get_device
from paths import CHECKPOINT_DIR, RESULTS_DIR
from profiler.snn_profiler import SNNProfiler
from models.metrics import accuracy
from models.training_setup import get_training_components


# ── Modello ──────────────────────────────────────────────────────────────────

class SpikingMLP(nn.Module):
    """SNN-MLP con neuroni LIF di SpikingJelly in multi-step mode (step_mode='m') e rate coding.

    Architettura per blocco:
        Linear → BatchNorm1d → LIFNode → Dropout

    L'input viene convertito in spike rate-coded ed elaborato vettorialmente
    lungo la dimensione temporale [num_steps, B, ...].
    L'output è costituito da spike e potenziale di membrana accumulati sull'ultimo layer LIF.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dims=(512, 256),
        beta=0.95,
        num_steps=25,
        dropout=0.2,
        direct_input=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.beta = beta
        self.direct_input = bool(direct_input) if direct_input is not None else False

        # Relazione tau e decadimento beta: tau = 1 / (1 - beta)
        tau = 1.0 / (1.0 - beta) if (beta is not None and beta < 1.0) else 20.0
        self.tau = tau

        # Surrogate gradient standard di SpikingJelly (ATan per stabilità numerica)
        spike_grad = surrogate.ATan()

        self.flatten = layer.Flatten(step_mode="m")

        # ── Layer nascosti ──
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lif_neurons = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(layer.Linear(in_dim, h_dim, bias=False, step_mode="m"))
            self.bns.append(layer.BatchNorm1d(h_dim, step_mode="m"))
            self.lif_neurons.append(
                neuron.ParametricLIFNode(
                    init_tau=tau,
                    decay_input=False,
                    surrogate_function=spike_grad,
                    step_mode="m",
                )
            )
            self.dropouts.append(layer.Dropout(p=dropout, step_mode="m"))
            in_dim = h_dim

        # ── Layer di output ──
        self.output_layer = layer.Linear(in_dim, num_classes, step_mode="m")
        self.output_lif = neuron.ParametricLIFNode(
            init_tau=tau,
            decay_input=False,
            surrogate_function=spike_grad,
            step_mode="m",
            store_v_seq=True,
        )

    def forward(self, x):
        """Forward pass multi-step con rate coding o direct input.

        Args:
            x: Tensor [B, C, H, W] o [B, D].

        Returns:
            spk_out: Spike di output accumulati [num_steps, B, num_classes]
            mem_out: Potenziale di membrana accumulato [num_steps, B, num_classes]
        """
        # Reset delle membrane all'inizio di ogni inferenza/passaggio
        functional.reset_net(self)

        # Propagazione temporale su num_steps timestep [num_steps, B, ...]
        x_seq = x.unsqueeze(0).repeat(self.num_steps, *([1] * x.ndim))
        if self.direct_input:
            cur = self.flatten(x_seq)
        else:
            rate = x_seq.clamp(0.0, 1.0)
            cur = torch.bernoulli(rate)
            cur = self.flatten(cur)

        # Propagazione multi-step attraverso i layer nascosti
        for linear, bn, lif, drop in zip(
            self.layers, self.bns, self.lif_neurons, self.dropouts
        ):
            cur = drop(lif(bn(linear(cur))))

        # Layer di output
        cur = self.output_layer(cur)
        spk_out = self.output_lif(cur)
        mem_out = self.output_lif.v_seq

        return spk_out, mem_out


# ── Factory ──────────────────────────────────────────────────────────────────

def build_snn_mlp(
    input_dim,
    num_classes,
    hidden_dims=(512, 256),
    beta=0.95,
    num_steps=25,
    dropout=0.2,
    direct_input=None,
):
    """Costruisce un SpikingMLP con i parametri specificati."""
    model = SpikingMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        beta=beta,
        num_steps=num_steps,
        dropout=dropout,
        direct_input=direct_input,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SNN-MLP layers: {[input_dim] + list(hidden_dims) + [num_classes]}")
    print(f"Timesteps: {num_steps} | Beta: {beta} (Tau: {model.tau:.2f}) | Dropout: {dropout} | DirectInput: {model.direct_input}")
    print(f"Parametri totali: {total_params:,}")
    return model


# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def snn_mlp_run(args, train_loader, test_loader, profiler=None, device=None):
    """Addestra e valuta un SNN-MLP con rate coding o direct input e SpikingJelly."""
    epochs = args.train.epochs
    weight_decay = getattr(args.train, "weight_decay", 1e-4)

    input_dim = args.model.input_dim
    num_classes = args.model.num_classes
    hidden_dims = tuple(args.model.hidden_dims)
    beta = getattr(args.model, "beta", 0.95)
    num_steps = getattr(args.model, "num_steps", 25)
    dropout = getattr(args.model, "dropout", 0.2)
    direct_input = getattr(args.model, "direct_input", None)

    device = get_device(device)
    if device.type == "mps":
        print("[SpikingJelly] I neuroni SpikingJelly supportano solo CPU e NVIDIA CUDA (non Apple Silicon MPS).")
        print("[SpikingJelly] L'esecuzione viene commutata automaticamente su CPU.")
        device = torch.device("cpu")

    model = build_snn_mlp(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        beta=beta,
        num_steps=num_steps,
        dropout=dropout,
        direct_input=direct_input,
    ).to(device)

    optimizer, criterion = get_training_components(
        model, args, model_type="snn_mlp"
    )
    snn_prof = SNNProfiler(model=model, tag="snn_mlp", output_dir=RESULTS_DIR)
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
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dims": list(hidden_dims),
            "beta": beta,
            "tau": model.tau,
            "num_steps": num_steps,
            "dropout": dropout,
            "backend": "spikingjelly",
        },
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model
