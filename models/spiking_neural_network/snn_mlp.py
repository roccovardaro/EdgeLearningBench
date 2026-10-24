"""
SNN-MLP (Spiking Neural Network – Multi-Layer Perceptron) con rate coding.

Utilizza neuroni Leaky Integrate-and-Fire (LIF) di snntorch, con rate coding
per la codifica degli input e surrogate gradient descent per l'addestramento.
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


# ── Modello ──────────────────────────────────────────────────────────────────

class SpikingMLP(nn.Module):
    """SNN-MLP con neuroni LIF e rate coding.

    Architettura per blocco:
        Linear → BatchNorm1d → LIF (Leaky Integrate-and-Fire)

    L'input viene presentato come spike rate-coded per `num_steps` timestep.
    L'output è la membrana cumulata sull'ultimo layer LIF.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dims=(512, 256),
        beta=0.95,
        num_steps=25,
        dropout=0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.beta = beta

        # Surrogate gradient: fast sigmoid
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # ── Layer nascosti ──
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lif_neurons = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, h_dim, bias=False))
            self.bns.append(nn.BatchNorm1d(h_dim))
            self.lif_neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True))
            self.dropouts.append(nn.Dropout(p=dropout))
            in_dim = h_dim

        # ── Layer di output ──
        self.output_layer = nn.Linear(in_dim, num_classes)
        self.output_lif = snn.Leaky(
            beta=beta, spike_grad=spike_grad, learn_beta=True, output=True
        )

    def forward(self, x):
        """Forward pass con rate coding.

        Args:
            x: Tensor [B, C, H, W] o [B, D].

        Returns:
            spk_out: Spike di output accumulati [num_steps, B, num_classes]
            mem_out: Potenziale di membrana accumulato [num_steps, B, num_classes]
        """
        x = x.flatten(1)  # [B, D]

        # ── Inizializza le membrane ──
        mem_list = []
        for lif in self.lif_neurons:
            mem_list.append(lif.init_leaky())
        mem_output = self.output_lif.init_leaky()

        # ── Registri per le uscite ──
        spk_recordings = []
        mem_recordings = []

        # ── Simulazione temporale con rate coding ──
        for step in range(self.num_steps):
            # Rate coding: gli input vengono convertiti in spike con probabilità
            # proporzionale all'intensità del pixel (dopo clamp in [0, 1]).
            rate = x.clamp(0.0, 1.0)
            inp = torch.bernoulli(rate)

            # Propagazione attraverso i layer nascosti
            cur = inp
            for i, (linear, bn, lif, drop) in enumerate(zip(self.layers, self.bns, self.lif_neurons, self.dropouts)):
                cur = linear(cur) #Dopo Linear otteniamo valori reali
                cur = bn(cur)
                spk, mem_list[i] = lif(cur, mem_list[i])
                cur = drop(spk)

            # Layer di output
            cur = self.output_layer(cur)
            spk_out, mem_output = self.output_lif(cur, mem_output)

            spk_recordings.append(spk_out)
            mem_recordings.append(mem_output)

        return torch.stack(spk_recordings), torch.stack(mem_recordings)


# ── Factory ──────────────────────────────────────────────────────────────────

def build_snn_mlp(
    input_dim,
    num_classes,
    hidden_dims=(512, 256),
    beta=0.95,
    num_steps=25,
    dropout=0.2,
):
    """Costruisce un SpikingMLP con i parametri specificati."""
    model = SpikingMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        beta=beta,
        num_steps=num_steps,
        dropout=dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SNN-MLP layers: {[input_dim] + list(hidden_dims) + [num_classes]}")
    print(f"Timesteps: {num_steps} | Beta: {beta} | Dropout: {dropout}")
    print(f"Parametri totali: {total_params:,}")
    return model


# SNNLoss è importata da training_setup



# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def snn_mlp_run(args, train_loader, test_loader, profiler=None, device=None):
    """Addestra e valuta un SNN-MLP con rate coding e kan surrogata."""
    epochs = args.train.epochs
    weight_decay = getattr(args.train, "weight_decay", 1e-4)

    input_dim = args.model.input_dim
    num_classes = args.model.num_classes
    hidden_dims = tuple(args.model.hidden_dims)
    beta = getattr(args.model, "beta", 0.95)
    num_steps = getattr(args.model, "num_steps", 25)
    dropout = getattr(args.model, "dropout", 0.2)

    mem_weight = getattr(args.train, "mem_weight", 1.0)
    spike_weight = getattr(args.train, "spike_weight", 1.0)

    device = get_device(device)

    model = build_snn_mlp(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        beta=beta,
        num_steps=num_steps,
        dropout=dropout,
    ).to(device)

    optimizer, criterion = get_training_components(
        model, args, model_type="snn_mlp"
    )
    snn_prof = SNNProfiler(model=model, tag="snn_mlp", output_dir=RESULTS_DIR)

    # ── Training ────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for  images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            spk_out, mem_out = model(images)
            loss = criterion(spk_out, mem_out, labels)


            loss.backward()
            # Gradient clipping per stabilità con surrogate gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            # Valutazione accuratezza
            mem_avg = mem_out.mean(dim=0).detach()
            correct += (mem_avg.argmax(1) == labels).sum().item()
            total += labels.size(0)
            
            snn_prof.update(spk_out)

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"loss={running_loss / total:.4f} | "
            f"train_acc={correct / total:.4f}"
        )

        if profiler:
            profiler.log_epoch(epoch + 1, running_loss / total, correct / total)
        snn_prof.log_epoch(epoch + 1)

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
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dims": list(hidden_dims),
            "beta": beta,
            "num_steps": num_steps,
            "dropout": dropout,
        },
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model
