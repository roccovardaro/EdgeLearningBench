"""
SNN-STDP (Spiking Neural Network – Spike-Timing Dependent Plasticity) con SpikingJelly.

Implementazione bio-ispirata per apprendimento non supervisionato:
- Neuroni Leaky Integrate-and-Fire con omeostasi intrinseca (AdaptiveLIFNode)
- Inibizione laterale ricorrente (PopulationCurrent) per competizione soft Winner-Take-All
- Plasticità locale STDP moltiplicativa (STDPLearner) con normalizzazione sinaptica
- Assegnazione classi per voting neuronale (Diehl & Cook 2015)
"""
import math
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import base, functional, layer, learning, neuron

from device import get_device
from paths import CHECKPOINT_DIR, RESULTS_DIR
from profiler.snn_profiler import SNNProfiler
from models.metrics import accuracy


# ── Componenti Neuronali ─────────────────────────────────────────────────────

class AdaptiveLIFNode(neuron.SimpleLIFNode):
    """Neurone LIF con omeostasi intrinseca (adattamento del bias) e refrattarietà locale."""

    def __init__(
        self,
        neurons: int,
        homeostasis_gain: float = 0.007775,
        v_threshold: float = 0.01,
        refractory_steps: int = 5,
        bias_initial: float = 0.01,
        num_steps: int = 25,
    ):
        super().__init__(tau=100.0, decay_input=True, v_threshold=v_threshold)
        bias = (
            -torch.rand(neurons) * bias_initial
            if bias_initial > 0
            else torch.zeros(neurons)
        )
        self.register_buffer("bias_current", bias)
        self.homeostasis_gain = homeostasis_gain
        self.target_rate = 120.0 / (neurons * num_steps)
        self.refractory_steps = refractory_steps
        self.register_memory("adaptation", 0.0)
        self.register_memory("refractory", 0.0)

    def neuronal_charge(self, x: torch.Tensor) -> None:
        super().neuronal_charge(x + self.bias_current - self.adaptation)
        self.v = torch.where(
            torch.as_tensor(self.refractory, device=x.device) > 0, 0.0, self.v
        )

    def neuronal_fire(self) -> torch.Tensor:
        return ((self.v >= self.v_threshold) & (torch.as_tensor(self.refractory, device=self.v.device) <= 0)).to(self.v)

    def neuronal_reset(self, spike: torch.Tensor) -> None:
        super().neuronal_reset(spike)
        prev_refr = torch.as_tensor(self.refractory, device=spike.device, dtype=spike.dtype)
        self.refractory = torch.where(spike.bool(), self.refractory_steps, (prev_refr - 1).clamp_min(0))
        self.adaptation = math.exp(-1.0 / 200.0) * self.adaptation + 0.01 * spike
        if self.training:
            self.bias_current.add_(self.target_rate - spike.mean(0), alpha=self.homeostasis_gain)


class CurrentFilter(base.MemoryModule):
    """Filtro esponenziale della corrente con costante temporale tau."""

    def __init__(self, tau: float = 5.0):
        super().__init__()
        self.tau = tau
        self.register_memory("current", 0.0)
        #states usa la corrente dal passo temporale precedente registrata in questo register memory

    def single_step_functional_forward(self, inputs, states, **kwargs):
        current = states[0] + (inputs[0] - states[0]) / self.tau
        return (current,), (current,)


class PopulationCurrent(nn.Module):
    """Inibizione laterale ricorrente e contrast enhancement di popolazione."""

    def __init__(self, neurons: int, inhibition: float = 0.09978):
        super().__init__()
        self.inhibition = inhibition
        self.register_buffer("input_gain", torch.ones(neurons))

    def forward(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """Applica inibizione laterale e contrast enhancement di popolazione."""
        if previous.shape != current.shape:
            raise ValueError("La dimensione del batch ricorrente è cambiata; reset prima dell'uso.")
        return self.input_gain * (
            current - current.mean(-1, keepdim=True)
        ) - self.inhibition * (previous.sum(-1, keepdim=True) - previous)


# ── Pre-elaborazione bio-ispirata (Difference of Gaussians) ──────────────────

class RetinalDoGFilter(nn.Module):
    """
    Filtro bio-ispirato Difference of Gaussians (DoG) che emula il filtraggio
    retinico ON-center / OFF-surround del sistema visivo biologico.
    Estrae i gradienti di contrasto ed elimina lo sfondo a bassa frequenza,
    garantendo sparsità ed evitando la saturazione sinaptica nell'STDP.
    """

    def __init__(self, size: int = 5, sigma1: float = 1.0, sigma2: float = 2.0):
        super().__init__()
        k = size // 2
        y, x = torch.meshgrid(torch.arange(-k, k + 1), torch.arange(-k, k + 1), indexing="ij")
        r2 = (x**2 + y**2).float()
        g1 = torch.exp(-r2 / (2.0 * sigma1**2)) / (2.0 * math.pi * sigma1**2)
        g2 = torch.exp(-r2 / (2.0 * sigma2**2)) / (2.0 * math.pi * sigma2**2)
        dog = g1 - g2
        dog = dog - dog.mean()
        self.register_buffer("kernel_1c", dog.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor, target_dim: Optional[int] = None) -> torch.Tensor:
        if x.dim() != 4:
            x_flat = x.flatten(1).float()
            if x_flat.min() < 0.0:
                x_min = x_flat.amin(dim=1, keepdim=True)
                x_max = x_flat.amax(dim=1, keepdim=True)
                return (x_flat - x_min) / (x_max - x_min).clamp_min(1e-6)
            if x_flat.max() > 1.0:
                x_flat = x_flat / 255.0
            return x_flat.clamp(0.0, 1.0)

        b, c, h, w = x.shape
        if c == 3 and target_dim is not None and target_dim == (h * w):
            x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
            c = 1

        k = self.kernel_1c.to(device=x.device, dtype=x.dtype)
        if c > 1:
            kernel = k.repeat(c, 1, 1, 1)
            filtered = F.conv2d(x, kernel, padding=k.shape[-1] // 2, groups=c)
        else:
            filtered = F.conv2d(x, k, padding=k.shape[-1] // 2)

        edges = F.relu(filtered)
        flat = edges.view(b, -1)
        max_val = flat.amax(dim=1, keepdim=True).clamp_min(1e-6)
        return (flat / max_val).clamp(0.0, 1.0)


# ── Utilità Input e Covarianza ───────────────────────────────────────────────

def prepare_input(
    x: torch.Tensor,
    dog_filter: Optional[RetinalDoGFilter] = None,
    target_dim: Optional[int] = None,
) -> torch.Tensor:
    """Standardizza l'input garantendo valori di intensità in [0, 1] e applicando DoG se presente."""
    if dog_filter is not None:
        return dog_filter(x, target_dim=target_dim)

    x_flat = x.flatten(1).float()
    if x_flat.max() > 1.0:
        x_flat = x_flat / 255.0
    return x_flat.clamp(0.0, 1.0)


def compute_dataset_covariance(
    loader,
    max_samples: int = 5000,
    device: str = "cpu",
    dog_filter: Optional[RetinalDoGFilter] = None,
    target_dim: Optional[int] = None,
) -> torch.Tensor:
    """Calcola la covarianza spaziale non supervisionata dei pixel per bilanciare il gain."""
    samples = []
    total = 0
    for imgs, _ in loader:
        samples.append(prepare_input(imgs, dog_filter=dog_filter, target_dim=target_dim).cpu())
        total += len(imgs)
        if total >= max_samples:
            break
    x = torch.cat(samples)[:max_samples].to(device)
    x = x - x.mean(dim=0)
    return (x.T @ x / (len(x) - 1)).clamp_min(1e-6)


# ── Modello SNN-STDP ─────────────────────────────────────────────────────────

class SpikingSTDP(nn.Module):
    """Rete SNN shallow con sinapsi Linear, inibizione ricorrente e voting neuronale."""

    def __init__(
        self,
        input_dim: int = 784,
        num_classes: int = 10,
        hidden_neurons: int = 128,
        num_steps: int = 25,
        homeostasis_gain: float = 0.007775,
        inhibition: float = 0.09978,
        v_threshold: float = 0.01,
        inhibition_tau: float = 5.0,
        refractory_steps: int = 5,
        bias_initial: float = 0.01,
        use_dog: bool = False,
        covariance: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_neurons = hidden_neurons
        self.num_steps = num_steps
        self.use_dog = use_dog
        self.dog_filter = RetinalDoGFilter() if use_dog else None

        # 1. Sinapsi lineare positiva e normalizzata
        self.synapse = nn.Linear(input_dim, hidden_neurons, bias=False)
        with torch.no_grad():
            self.synapse.weight.uniform_(0.0, 1.0)
            target_sum = 0.1 * input_dim
            self.synapse.weight.mul_(target_sum / self.synapse.weight.sum(1, keepdim=True).clamp_min(1e-12))

        # 2. Modulo neurale ricorrente: CurrentFilter + AdaptiveLIFNode + PopulationCurrent
        self.filter = CurrentFilter(inhibition_tau)
        self.lif = AdaptiveLIFNode(
            hidden_neurons,
            homeostasis_gain=homeostasis_gain,
            v_threshold=v_threshold,
            refractory_steps=refractory_steps,
            bias_initial=bias_initial,
            num_steps=num_steps,
        )
        self.pop_current = PopulationCurrent(hidden_neurons, inhibition)
        self.sn = layer.ElementWiseRecurrentContainer(
            nn.Sequential(self.filter, self.lif),
            self.pop_current,
        )

        self.requires_grad_(False)
        self.to(device)

        # 3. Covarianza dei pixel e bilanciamento input gain
        if covariance is None:
            covariance = torch.eye(input_dim, device=device)
        self.register_buffer("pixel_covariance", covariance.to(device=device, dtype=self.synapse.weight.dtype))
        self.register_buffer("reference_input_variance", self.input_variance().clone())
        self.refresh_gain()

        # 4. Assegnazione classi e pesi di specificità per il voting (-1 = muto)
        self.register_buffer("assigned_labels", torch.full((hidden_neurons,), -1, dtype=torch.long))
        self.register_buffer("assigned_weights", torch.ones((hidden_neurons,), dtype=torch.float32))

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Standardizza o filtra l'input prima dell'encoding a impulsi."""
        return prepare_input(x, dog_filter=self.dog_filter, target_dim=self.input_dim)

    @torch.no_grad()
    def input_variance(self) -> torch.Tensor:
        w = self.synapse.weight
        field = w - w.mean(dim=0, keepdim=True)
        return (field @ self.pixel_covariance * field).sum(dim=1)

    @torch.no_grad()
    def refresh_gain(self) -> None:
        variance = self.input_variance().clamp_min(1e-6)
        self.sn.element_wise_function.input_gain.copy_(
            (self.reference_input_variance / variance).clamp(0.1, 10.0).sqrt()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrazione temporale standard per inferenza (conforme a SpikingMLP)."""
        functional.reset_net(self)
        dev = self.synapse.weight.device
        x_flat = self.prepare_input(x).to(dev)
        prob_scale = min(0.8, max(0.01, 45.0 / self.num_steps))
        probability = x_flat * prob_scale

        encoder = neuron.SimpleIFNode(v_threshold=1.0, v_reset=None).to(dev)
        encoder.v = torch.rand(probability.shape, device=dev)

        spk_records, mem_records = [], []
        for _ in range(self.num_steps):
            in_spk = encoder(probability)
            current = self.synapse(in_spk)
            out_spk = self.sn(current)
            spk_records.append(out_spk)
            mem_records.append(self.lif.v)

        return torch.stack(spk_records), torch.stack(mem_records)

    def forward_and_update(self, x: torch.Tensor, learner: learning.STDPLearner) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrazione temporale con aggiornamento locale STDP a ogni timestep."""
        functional.reset_net(self)
        learner.reset()
        dev = self.synapse.weight.device

        # Costruzione corrente costante per ciascun pixel
        x_flat = self.prepare_input(x).to(dev)
        prob_scale = min(0.8, max(0.01, 45.0 / self.num_steps))
        probability = x_flat * prob_scale

        #Trasforma i pixel in inpulsi
        encoder = neuron.SimpleIFNode(v_threshold=1.0, v_reset=None).to(dev)
        encoder.v = torch.rand(probability.shape, device=dev) #potenziale iniziale è casuale

        spk_records, mem_records = [], []
        for _ in range(self.num_steps):
            in_spk = encoder(probability)
            current = self.synapse(in_spk)
            out_spk = self.sn(current) # elabora la corrente e produce gli spike
            spk_records.append(out_spk)
            mem_records.append(self.lif.v)

            # Aggiornamento STDP online con normalizzazione sinaptica
            dw = learner.step(on_grad=False, scale=1.0 / len(x))
            weight = self.synapse.weight
            weight.add_(dw).clamp_(0.0, 1.0)
            target_sum = 0.1 * self.input_dim
            weight.mul_(target_sum / weight.sum(1, keepdim=True).clamp_min(1e-12))
            weight.clamp_(0.0, 1.0) # forza tutti i pesi nell'intervallo [0,1]

        self.refresh_gain()
        learner.reset()
        functional.reset_net(self)

        return torch.stack(spk_records), torch.stack(mem_records)

    def _predict_counts(self, counts: torch.Tensor) -> torch.Tensor:
        """Predice la classe ponderando i voti in base alla specificità di ciascun neurone (Diehl & Cook 2015)."""
        scores = counts.new_zeros((counts.shape[0], self.num_classes))
        assigned_labels = self.assigned_labels.to(counts.device)
        assigned_weights = self.assigned_weights.to(counts.device)
        for label in range(self.num_classes):
            selected = assigned_labels == label
            if selected.any():
                weights = assigned_weights[selected]
                scores[:, label] = (counts[:, selected] * weights).sum(dim=1) / weights.sum().clamp_min(1e-6)
        preds = scores.argmax(dim=1)
        silent = scores.max(dim=1).values == 0
        if silent.any():
            preds[silent] = -1
        return preds

    @torch.no_grad()
    def assign_labels(self, loader, max_samples: int = 5000, device=None) -> Tuple[torch.Tensor, float]:
        """Assegna a ogni neurone la classe e la specificità di voting, calcolando l'accuratezza."""
        self.eval()
        dev = device or next(self.parameters()).device
        counts_list, labels_list = [], []
        total = 0

        for imgs, lbls in loader:
            imgs = imgs.to(dev)
            lbls = lbls.to(dev)
            spk_out, _ = self.forward(imgs) # [T,B,N]
            counts_list.append(spk_out.sum(dim=0)) # [B,N]
            labels_list.append(lbls)
            total += len(imgs)
            if total >= max_samples: # Interrompe la raccolta quando raggiunge abbastanza immagini (max_samples)
                break

        counts = torch.cat(counts_list)[:max_samples]
        labels = torch.cat(labels_list)[:max_samples]

        means = torch.stack(
            [
                counts[labels == c].sum(dim=0) / (labels == c).sum().clamp_min(1)
                for c in range(self.num_classes)
            ]
        )
        assignment = means.argmax(dim=0)
        assignment = assignment.masked_fill(means.max(dim=0).values == 0, -1)
        self.assigned_labels.copy_(assignment)

        # Calcolo della specificità neuronale (Diehl & Cook 2015):
        # Un neurone selettivo ha un peso alto, un neurone con risposta generica ha peso basso
        norm_means = means / means.sum(dim=0, keepdim=True).clamp_min(1e-6)
        weights = torch.zeros(self.hidden_neurons, device=dev)
        for i in range(self.hidden_neurons):
            c = assignment[i].item()
            if c >= 0:
                weights[i] = norm_means[c, i]
        self.assigned_weights.copy_(weights)

        # Calcolo immediato dell'accuratezza sul set di calibrazione con voting ponderato
        preds = self._predict_counts(counts)
        acc = (preds == labels).float().mean().item()
        return assignment, acc

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predice le etichette di classe per voting neuronale (usato da models.metrics.accuracy)."""
        was_training = self.training
        self.eval()
        try:
            spk_out, _ = self.forward(x)
            counts = spk_out.sum(dim=0)  # [B, hidden_neurons]
            return self._predict_counts(counts)
        finally:
            self.train(was_training)


# Alias per retrocompatibilità
SpikingSTDPNet = SpikingSTDP


# ── Factory ──────────────────────────────────────────────────────────────────

def build_snn_stdp(
    input_dim: int = 784,
    num_classes: int = 10,
    hidden_neurons: int = 128,
    num_steps: int = 25,
    homeostasis_gain: float = 0.007775,
    inhibition: float = 0.09978,
    v_threshold: float = 0.01,
    inhibition_tau: float = 5.0,
    refractory_steps: int = 5,
    bias_initial: float = 0.01,
    use_dog: bool = False,
    covariance: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> SpikingSTDP:
    """Costruisce un SpikingSTDP con i parametri specificati (conforme a build_snn_mlp)."""
    model = SpikingSTDP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_neurons=hidden_neurons,
        num_steps=num_steps,
        homeostasis_gain=homeostasis_gain,
        inhibition=inhibition,
        v_threshold=v_threshold,
        inhibition_tau=inhibition_tau,
        refractory_steps=refractory_steps,
        bias_initial=bias_initial,
        use_dog=use_dog,
        covariance=covariance,
        device=device,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SNN-STDP layers: [{input_dim} -> {hidden_neurons} (AdaptiveLIF) -> {num_classes} (Voting)]")
    print(f"Timesteps: {num_steps} | V_th: {v_threshold} | Inibizione: {inhibition} | Omeostasi: {homeostasis_gain} | DoG: {use_dog}")
    print(f"Parametri totali: {total_params:,}")
    return model


# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def snn_stdp_run(args, train_loader, test_loader, profiler=None, device=None) -> SpikingSTDP:
    """Addestra e valuta un SNN con STDP (conforme a snn_mlp_run)."""
    epochs = args.train.epochs
    batch_size = getattr(args.train, "batch_size", 128)
    ltp = getattr(args.train, "ltp", 0.002296)
    train_samples = getattr(args.train, "train_samples", None)
    calib_samples = getattr(args.train, "calibration_samples", 5000)

    input_dim = args.model.input_dim
    num_classes = getattr(args.model, "num_classes", 10)
    hidden_neurons = getattr(args.model, "hidden_neurons", 128)
    num_steps = getattr(args.model, "num_steps", 25)
    homeostasis_gain = getattr(args.model, "homeostasis_gain", 0.007775)
    inhibition = getattr(args.model, "inhibition", 0.09978)
    v_threshold = getattr(args.model, "v_threshold", 0.01)
    inhibition_tau = getattr(args.model, "inhibition_tau", 5.0)
    refractory_steps = getattr(args.model, "refractory_steps", 5)
    bias_initial = getattr(args.model, "bias_initial", 0.01)
    use_dog = getattr(args.model, "use_dog", False)

    device = get_device(device)

    # 1. Filtro DoG preliminare per calcolo covarianza
    dog_filter = RetinalDoGFilter() if use_dog else None

    # Calcolo covarianza pixel non supervisionata
    covariance = compute_dataset_covariance(
        train_loader,
        max_samples=5000,
        device=device,
        dog_filter=dog_filter,
        target_dim=input_dim,
    )

    # 2. Costruzione modello
    model = build_snn_stdp(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_neurons=hidden_neurons,
        num_steps=num_steps,
        homeostasis_gain=homeostasis_gain,
        inhibition=inhibition,
        v_threshold=v_threshold,
        inhibition_tau=inhibition_tau,
        refractory_steps=refractory_steps,
        bias_initial=bias_initial,
        use_dog=use_dog,
        covariance=covariance,
        device=device,
    )

    # 3. Learner STDP locale di SpikingJelly (disabilitato inizialmente per evitare record spuri)
    learner = learning.STDPLearner(
        "s",
        model.synapse,
        model.sn,
        tau_pre=20.0,
        tau_post=20.0,
        f_pre=lambda w: ltp * w,
        f_post=lambda w: ltp * (1.0 - w),
    )
    learner.disable()
    learner.reset()

    snn_prof = SNNProfiler(model=model, tag="snn_stdp", output_dir=RESULTS_DIR)
    budget_tracker = getattr(args, "budget_tracker", getattr(profiler, "budget_tracker", None))
    learner.reset()
    initial_weight = model.synapse.weight.clone()

    # ── Training (STDP non supervisionato) ───────────────────────
    budget_stop = False
    train_acc = 0.0
    silent_pct = 0.0
    try:
        for epoch in range(epochs):
            model.train()
            learner.enable()
            total_processed = 0

            for images, _ in train_loader:
                images = images.to(device)

                if budget_tracker and budget_tracker.static_fwd_macs == 0:
                    budget_tracker.attach_model(model, sample_input=images[:1], device=device)
                snn_prof.ensure_static_macs(images[:1])

                spk_out, _ = model.forward_and_update(images, learner=learner)

                total_processed += len(images)
                snn_prof.update(spk_out)

                batch_fr = spk_out.sum().item() / spk_out.numel() if spk_out.numel() > 0 else 0.0
                if budget_tracker and budget_tracker.step(
                    batch_size=len(images),
                    firing_rate=batch_fr,
                    num_steps_snn=num_steps,
                    epoch=epoch + 1,
                ):
                    budget_stop = True
                    break

                if train_samples and total_processed >= train_samples:
                    break

            learner.disable()

            # Assegnazione etichette ai neuroni ogni 2 epoche o all'ultima epoca / stop del budget
            if ((epoch + 1) % 2 == 0) or (epoch + 1 == epochs) or budget_stop:
                assignment, train_acc = model.assign_labels(train_loader, max_samples=calib_samples, device=device)
                silent_pct = (assignment == -1).float().mean().item() * 100.0

            weight_change = (model.synapse.weight - initial_weight).abs().mean().item()

            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"train_acc={train_acc:.4f} | "
                f"Δw={weight_change:.6f} | "
                f"silent={silent_pct:.1f}%"
            )

            profiler.log_epoch(epoch + 1, 0.0, train_acc)
            snn_prof.log_epoch(epoch + 1)

            if budget_stop:
                break

        # ── Valutazione ─────────────────────────────────────────────
        args.edge_profiler.stop()

        snn_prof.reset()
        test_acc = accuracy(model, test_loader, device=device)
        snn_prof.log_epoch("test", tag="test")

        print(f"Train accuracy: {train_acc:.4f}  (error: {1 - train_acc:.4f})")
        print(f"Test  accuracy: {test_acc:.4f}  (error: {1 - test_acc:.4f})")

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
                "hidden_neurons": hidden_neurons,
                "num_steps": num_steps,
                "homeostasis_gain": homeostasis_gain,
                "inhibition": inhibition,
                "v_threshold": v_threshold,
                "inhibition_tau": inhibition_tau,
                "refractory_steps": refractory_steps,
                "bias_initial": bias_initial,
                "use_dog": use_dog,
                "backend": "spikingjelly_stdp",
            },
            "assigned_labels": model.assigned_labels.cpu(),
            "assigned_weights": model.assigned_weights.cpu(),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        }, checkpoint_path)
        print(f"Checkpoint salvato in: {checkpoint_path}")

    finally:
        learner.close()

    return model
