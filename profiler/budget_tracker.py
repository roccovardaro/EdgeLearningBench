"""
BudgetTracker: Modulo per il tracciamento e controllo del budget computazionale fisso
in termini di MAC equivalenti (MAC_eq).

Consente un confronto fair, riproducibile ed energeticamente consistente tra paradigmi
eterogenei (ANN standard con backprop, SNN, Forward-Forward, Hebbiano/CHL, KAN, LTC).

Modello di Costo Computazionale:
  1. Forward pass (C_fwd):
     - ANN / Dense / Conv: calcolo statico del numero di moltiplicazioni e accumuli per campione * batch_size.
     - SNN: Synaptic Operations (SOP) sparse moltiplicate per il fattore di conversione energetico
            o operazionale: SOP * alpha_sop * batch_size + dinamica di membrana neuronale.
     - KAN: valutazione di B-splines per connessione (griglia G + ordine k + 1) * Linear MACs.
     - CHL: (T_free + T_clamped) passi di rilassamento ricorrente * forward pass.
     - Forward-Forward: 2 forward pass (dati positivi e negativi) * batch_size.
     - LTC / ODE: forward pass proiettato su sequenze temporali con integrazione dinamica.

  2. Backward pass (C_bwd):
     - ANN standard (Backprop): C_bwd ≈ 2 * C_fwd (gradiente attivazioni + gradiente pesi).
     - SNN (BPTT con Surrogate Gradient): gradienti continui densi attraverso T intervalli temporali:
       C_bwd ≈ 2 * T * C_fwd_dense.
     - Forward-Forward: gradienti locali dei singoli layer rispetto alla bontà: C_bwd ≈ 2 * C_fwd_layer.
     - CHL: nessun backward con gradiente; aggiornamento contrastivo locale delle co-attivazioni: C_bwd ≈ 1 * C_fwd.
     - KAN: autograd continuo sulle B-spline: C_bwd ≈ 2 * C_fwd_kan.

  3. Aggiornamento pesi dell'ottimizzatore (C_opt):
     - SGD: 1 moltiplicazione scalare + 1 somma per parametro => 0.5 MAC / parametro.
     - Adam / AdamW: calcolo del primo e secondo momento, bias correction, radice e divisione:
       ~10 FLOP / parametro => ~2.5 MAC_eq / parametro per batch.

Metrica di conversione:
  - Energy Mode (Horowitz 2014, CMOS 45nm @ FP32/16):
      E_MAC = 4.6 pJ, E_AC = 0.9 pJ => 1 SOP (AC) = 0.1956 MAC_eq (circa 0.20 MAC).
  - Operational Mode (FLOP count puro):
      1 MAC = 2 FLOPs, 1 AC = 1 FLOP => 1 SOP (AC) = 0.50 MAC_eq.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class BudgetExceededException(Exception):
    """Eccezione sollevata quando il budget computazionale massimo è stato superato."""
    pass


def compute_model_static_macs(
    model: nn.Module,
    sample_input: Optional[torch.Tensor] = None,
    input_dim: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> int:
    """
    Calcola staticamente il numero teorico di MAC per un singolo campione nel forward pass.
    Supporta Linear, Conv2d, BatchNorm, KAN (KANLinear), e layer SpikingJelly.
    """
    macs_dict: Dict[str, int] = {}
    hooks = []

    def make_hook(name):
        def hook(mod, inp, out):
            mod_class = mod.__class__.__name__

            # 1. Layer Convoluzionali
            if isinstance(mod, nn.Conv2d):
                out_tensor = out[0] if isinstance(out, tuple) else out
                kh, kw = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
                cin = mod.in_channels // mod.groups
                h_out, w_out = out_tensor.shape[2], out_tensor.shape[3]
                cout = mod.out_channels
                macs_dict[name] = int(cin * kh * kw * cout * h_out * w_out)

            # 2. Layer KAN (efficient-kan KANLinear)
            elif "KANLinear" in mod_class:
                in_f = getattr(mod, "in_features", 0)
                out_f = getattr(mod, "out_features", 0)
                grid_s = getattr(mod, "grid_size", 5)
                spline_o = getattr(mod, "spline_order", 3)
                num_bases = grid_s + spline_o
                # Base lineare + combinazione spline pesata
                macs_dict[name] = int(in_f * out_f * (1 + num_bases))

            # 3. Layer Lineari standard o SpikingJelly layer.Linear
            elif isinstance(mod, nn.Linear) or hasattr(mod, "in_features") and hasattr(mod, "out_features"):
                in_f = getattr(mod, "in_features", 0)
                out_f = getattr(mod, "out_features", 0)
                macs_dict[name] = int(in_f * out_f)

        return hook

    # Registra hook sui moduli computazionali (Linear, Conv2d, KANLinear)
    for name, m in model.named_modules():
        mod_class = m.__class__.__name__
        if isinstance(m, (nn.Linear, nn.Conv2d)) or "KANLinear" in mod_class:
            hooks.append(m.register_forward_hook(make_hook(name)))

    dev = device or next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    was_training = model.training
    model.eval()

    # Prepara un dummy input rappresentativo per triggerare gli hook
    dummy = sample_input
    if dummy is None:
        if input_dim is not None:
            if isinstance(input_dim, (int, float)):
                dummy = torch.zeros(1, int(input_dim), device=dev)
            elif isinstance(input_dim, (tuple, list)):
                dummy = torch.zeros(1, *input_dim, device=dev)
        elif getattr(model, "input_dim", None) is not None:
            inp_d = model.input_dim
            if isinstance(inp_d, (int, float)):
                dummy = torch.zeros(1, int(inp_d), device=dev)
            elif isinstance(inp_d, (tuple, list)):
                dummy = torch.zeros(1, *inp_d, device=dev)
        else:
            # Ricerca euristica da layer Conv o Linear
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    dummy = torch.zeros(1, m.in_channels, 32, 32, device=dev)
                    break
                elif isinstance(m, nn.Linear):
                    dummy = torch.zeros(1, m.in_features, device=dev)
                    break
            if dummy is None:
                dummy = torch.zeros(1, 784, device=dev)

    # Assicura batch size pari a 1 per il dummy test
    if dummy.shape[0] != 1:
        dummy = dummy[:1]

    old_steps = getattr(model, "num_steps", None)
    if old_steps is not None:
        model.num_steps = 1

    try:
        with torch.no_grad():
            model(dummy.to(dev))
    except Exception:
        # Fallback per reti con firme speciali (es. CHL o modelli ricorrenti)
        pass
    finally:
        if old_steps is not None:
            model.num_steps = old_steps
        if was_training:
            model.train()
        for h in hooks:
            h.remove()

    total_macs = sum(macs_dict.values())

    # Fallback analitico se gli hook non hanno catturato operazioni (es. CHL o tensori custom)
    if total_macs == 0:
        for p_name, p in model.named_parameters():
            if p.ndim == 2:  # Matrice pesi [out_features, in_features]
                total_macs += p.numel()
            elif p.ndim == 4:  # Kernel conv [out_c, in_c, kh, kw]
                total_macs += p.numel() * 28 * 28  # Stima risoluzione media

    return max(1, total_macs)


class BudgetTracker:
    """
    Traccia ed impone un budget computazionale fisso durante l'addestramento.
    Supporta tutti i paradigmi di calcolo ed esporta report dettagliati.
    Adotta la metrica operazionale standard (Op-Count): 1 SOP (AC) = 0.50 MAC_eq.
    """

    ALPHA_SOP = 0.50  # 1 AC = 1 FLOP = 0.50 MAC

    def __init__(
        self,
        target_macs: Optional[Union[float, int, str]] = None,
        enabled: bool = True,
        paradigm: str = "ann",
        include_optimizer: bool = True,
        optimizer_type: str = "adamw",
        tag: str = "benchmark",
        output_dir: Optional[Union[str, Path]] = None,
    ):
        if target_macs is not None:
            try:
                self.target_macs = float(target_macs)
            except (ValueError, TypeError):
                self.target_macs = None
        else:
            self.target_macs = None

        self.enabled = bool(enabled and self.target_macs is not None and self.target_macs > 0)
        self.paradigm = (paradigm or "ann").lower()
        self.include_optimizer = bool(include_optimizer)
        self.optimizer_type = (optimizer_type or "adamw").lower()
        self.tag = tag
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "results"

        # Metriche statiche del modello (inizializzate con attach_model)
        self.static_fwd_macs: int = 0
        self.num_params: int = 0
        self.num_neurons: int = 0

        # Accumulatori di progresso computazionale
        self.cum_macs: float = 0.0
        self.cum_fwd_macs: float = 0.0
        self.cum_bwd_macs: float = 0.0
        self.cum_opt_macs: float = 0.0
        self.samples_seen: int = 0
        self.steps: int = 0
        self.exhausted: bool = False
        self.stop_step: Optional[int] = None
        self.stop_epoch: Optional[int] = None

        # Storico per step
        self.step_history: List[Dict[str, Any]] = []

    def attach_model(
        self,
        model: nn.Module,
        sample_input: Optional[torch.Tensor] = None,
        input_dim: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Collega il modello e ne calcola i parametri e i MAC statici per singolo campione."""
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.num_params == 0:
            self.num_params = sum(p.numel() for p in model.parameters())

        self.static_fwd_macs = compute_model_static_macs(
            model=model,
            sample_input=sample_input,
            input_dim=input_dim,
            device=device,
        )

        # Stima del numero di neuroni (utile per la dinamica di stato SNN/LTC)
        neurons = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                neurons += m.out_features
        self.num_neurons = max(1, neurons)

        status_str = f"ATTIVO (Target: {self.target_macs:.2e} MAC_eq)" if self.enabled else "PASSIVO (Solo monitoraggio)"
        print(f"[BudgetTracker] Inizializzato per paradigma '{self.paradigm}' [{status_str}]")
        print(f"[BudgetTracker] Parametri: {self.num_params:,} | MAC statici/campione forward: {self.static_fwd_macs:,}")

    def compute_step_macs(
        self,
        batch_size: int,
        firing_rate: float = 0.0,
        num_steps_snn: int = 1,
        chl_steps: Optional[int] = None,
        is_ff_step: bool = False,
    ) -> Dict[str, float]:
        """
        Calcola la ripartizione esatta dei MAC equivalenti per il batch corrente.
        Restituisce un dizionario con 'fwd', 'bwd', 'opt' e 'total'.
        """
        # 1. Costo di aggiornamento dei pesi (C_opt)
        if not self.include_optimizer or "chl" in self.paradigm:
            c_opt = 0.0
        elif "adam" in self.optimizer_type:
            # AdamW: ~10 FLOP / parametro => ~2.5 MAC_eq
            c_opt = 2.5 * self.num_params
        else:
            # SGD con momentum / standard: ~2-4 FLOP => ~0.5 - 1.0 MAC_eq
            c_opt = 0.5 * self.num_params

        # 2. Costo del Forward pass (C_fwd) e Backward pass (C_bwd)
        p = self.paradigm

        if "snn" in p:
            # SNN Forward: accumuli sparsi (SOP) convertiti via ALPHA_SOP (1 SOP = 0.5 MAC)
            total_sops = firing_rate * num_steps_snn * self.static_fwd_macs * batch_size
            c_fwd_synaptic = total_sops * self.ALPHA_SOP
            # Dinamica di membrana dei neuroni (leak + reset): ~2 FLOPs = 1 MAC per neurone per time-step
            c_fwd_soma = (self.num_neurons * num_steps_snn * batch_size) * self.ALPHA_SOP
            c_fwd = c_fwd_synaptic + c_fwd_soma

            # SNN Backward con BPTT e Surrogate Gradient:
            # L'autograd retropropaga tensori densi in virgola mobile lungo tutti i T step temporali
            c_bwd = 2.0 * (self.static_fwd_macs * batch_size) * num_steps_snn

        elif "ff" in p or is_ff_step:
            # Forward-Forward: 2 passaggi per batch (positivo + negativo)
            c_fwd = 2.0 * self.static_fwd_macs * batch_size
            # Calcolo locale della goodness e gradienti dei singoli pesi (senza autograd globale)
            c_bwd = 2.0 * self.static_fwd_macs * batch_size

        elif "chl" in p:
            # Contrastive Hebbian: rilassamento ricorrente (T_free + T_clamped)
            steps_recurrent = chl_steps if chl_steps is not None else 50
            c_fwd = steps_recurrent * self.static_fwd_macs * batch_size
            # Aggiornamento locale Hebbiano delle covarianze tra attivazioni (prodotto esterno)
            c_bwd = 1.0 * self.static_fwd_macs * batch_size

        elif "kan" in p:
            # KAN: le B-splines sono già computate in self.static_fwd_macs
            c_fwd = self.static_fwd_macs * batch_size
            c_bwd = 2.0 * c_fwd

        elif "ltc" in p:
            # LTC / Neural ODE: forward con sequenza temporale (default 16 steps)
            t_ltc = getattr(self, "ltc_steps", 16)
            c_fwd = self.static_fwd_macs * batch_size
            c_bwd = 2.0 * c_fwd

        else:
            # Baseline ANN standard (MLP, CNN, ResNet con Backpropagation)
            c_fwd = self.static_fwd_macs * batch_size
            c_bwd = 2.0 * c_fwd

        c_total = c_fwd + c_bwd + c_opt
        return {"fwd": c_fwd, "bwd": c_bwd, "opt": c_opt, "total": c_total}

    def step(
        self,
        batch_size: int,
        firing_rate: float = 0.0,
        num_steps_snn: int = 1,
        chl_steps: Optional[int] = None,
        is_ff_step: bool = False,
        epoch: int = 1,
        loss: Optional[float] = None,
        acc: Optional[float] = None,
    ) -> bool:
        """
        Registra un batch di addestramento.
        Restituisce True se il budget è stato raggiunto/superato (richiesta di interruzione).
        """
        costs = self.compute_step_macs(
            batch_size=batch_size,
            firing_rate=firing_rate,
            num_steps_snn=num_steps_snn,
            chl_steps=chl_steps,
            is_ff_step=is_ff_step,
        )

        self.cum_fwd_macs += costs["fwd"]
        self.cum_bwd_macs += costs["bwd"]
        self.cum_opt_macs += costs["opt"]
        self.cum_macs += costs["total"]
        self.samples_seen += batch_size
        self.steps += 1

        # Controllo raggiungimento budget
        if self.enabled and self.target_macs is not None and self.cum_macs >= self.target_macs:
            if not self.exhausted:
                self.exhausted = True
                self.stop_step = self.steps
                self.stop_epoch = epoch
                pct = (self.cum_macs / self.target_macs) * 100.0
                print("\n" + "!" * 70)
                print(f"[BudgetTracker] 🛑 BUDGET COMPUTAZIONALE RAGGIUNTO!")
                print(f"[BudgetTracker] MAC_eq spesi: {self.cum_macs:.3e} / {self.target_macs:.3e} ({pct:.1f}%)")
                print(f"[BudgetTracker] Campioni processati: {self.samples_seen:,} | Step: {self.steps} (Epoca: {epoch})")
                print("!" * 70 + "\n")
            return True

        return False

    @property
    def is_exhausted(self) -> bool:
        """Restituisce True se il budget è stato superato."""
        return self.exhausted

    def print_summary(self) -> None:
        """Stampa un riepilogo scientifico ed elegante del costo di addestramento."""
        total = max(1.0, self.cum_macs)
        pct_fwd = (self.cum_fwd_macs / total) * 100.0
        pct_bwd = (self.cum_bwd_macs / total) * 100.0
        pct_opt = (self.cum_opt_macs / total) * 100.0

        print("\n" + "=" * 65)
        print("          RIEPILOGO BUDGET COMPUTAZIONALE (MAC_eq)")
        print("=" * 65)
        print(f"  Paradigma di calcolo:       {self.paradigm.upper()}")
        print(f"  Stato del vincolo:          {'RAGGIUNTO (Stop)' if self.exhausted else ('IN CORSO' if self.enabled else 'DISATTIVATO')}")
        if self.target_macs:
            print(f"  Budget prefissato:          {self.target_macs:.3e} MAC_eq")
        print(f"  MAC_eq totali accumulati:   {self.cum_macs:.3e} MAC_eq")
        print(f"  Campioni di training visti: {self.samples_seen:,}")
        print(f"  Step di ottimizzazione:     {self.steps:,}")
        print("-" * 65)
        print("  Ripartizione del Costo di Addestramento:")
        print(f"    - Forward pass (C_fwd):   {self.cum_fwd_macs:.3e} MAC_eq  ({pct_fwd:5.1f}%)")
        print(f"    - Backward pass (C_bwd):  {self.cum_bwd_macs:.3e} MAC_eq  ({pct_bwd:5.1f}%)")
        print(f"    - Weight update (C_opt):  {self.cum_opt_macs:.3e} MAC_eq  ({pct_opt:5.1f}%)")
        print("=" * 65 + "\n")

    def export_csv(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Esporta il riepilogo finale del budget in CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = Path(filepath) if filepath else self.output_dir / f"{self.tag}_budget_summary.csv"

        fieldnames = [
            "tag", "paradigm", "target_macs", "cum_macs", "cum_fwd_macs",
            "cum_bwd_macs", "cum_opt_macs", "samples_seen", "steps",
            "budget_exhausted"
        ]
        row = {
            "tag": self.tag,
            "paradigm": self.paradigm,
            "target_macs": self.target_macs,
            "cum_macs": round(self.cum_macs, 2),
            "cum_fwd_macs": round(self.cum_fwd_macs, 2),
            "cum_bwd_macs": round(self.cum_bwd_macs, 2),
            "cum_opt_macs": round(self.cum_opt_macs, 2),
            "samples_seen": self.samples_seen,
            "steps": self.steps,
            "budget_exhausted": self.exhausted,
        }

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

        print(f"[BudgetTracker] Riepilogo budget esportato in: {path}")
        return path


def get_budget_tracker(
    args: Any,
    model: Optional[nn.Module] = None,
    paradigm: Optional[str] = None,
    device: Optional[torch.device] = None,
    sample_input: Optional[torch.Tensor] = None,
    profiler: Optional[Any] = None,
) -> Optional[BudgetTracker]:
    """
    Costruttore unificato: estrae i parametri di configurazione da `args`,
    crea o recupera il `BudgetTracker` e vi collega il modello.
    """
    # 1. Se il profiler contiene già un'istanza, riutilizzala
    if profiler is not None and getattr(profiler, "budget_tracker", None) is not None:
        tracker = profiler.budget_tracker
        if model is not None and tracker.static_fwd_macs == 0:
            tracker.attach_model(model, sample_input=sample_input, device=device)
        return tracker

    # 2. Estrazione configurazione da args.budget
    b_cfg = getattr(args, "budget", None)
    enabled = getattr(b_cfg, "enabled", False) if b_cfg is not None else False
    max_macs = getattr(b_cfg, "max_macs", None) if b_cfg is not None else None
    inc_opt = getattr(b_cfg, "include_optimizer", True) if b_cfg is not None else True

    tag = f"{paradigm or getattr(args, 'paradigm', 'model')}"
    if hasattr(args, "dataset") and hasattr(args.dataset, "name"):
        tag += f"_{args.dataset.name}"

    out_dir = getattr(args, "results_dir", None)

    tracker = BudgetTracker(
        target_macs=max_macs,
        enabled=enabled,
        paradigm=paradigm or getattr(args, "paradigm", "ann"),
        include_optimizer=inc_opt,
        tag=tag,
        output_dir=out_dir,
    )

    if model is not None:
        tracker.attach_model(model, sample_input=sample_input, device=device)

    # Collega al profiler se disponibile
    if profiler is not None:
        profiler.budget_tracker = tracker

    return tracker

