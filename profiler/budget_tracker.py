"""
BudgetTracker: Modulo per il tracciamento e controllo del budget computazionale fisso.
Definito come "Budget Sinaptico Teorico Stimato in MAC Equivalenti (MAC_eq)"
ad ambito parziale: consideriamo le operazioni sinaptiche e di correlazione,
escludendo le operazioni accessorie e l'aggiornamento pesi dell'ottimizzatore.

Convenzione e presupposti dichiarati (agnostici dall'hardware):
  - 1 MAC = 1 MUL + 1 ADD = 2 FLOPs
  - 1 AC (Accumulo sinaptico puro / Addizione) = 0.50 MAC (1 FLOP)
  - Il conteggio sparso delle SNN presuppone hardware neuromorfico o algoritmi event-driven
    capaci di saltare le operazioni associate agli spike assenti (zeri).
  - ANN standard (Backprop): C_fwd = 1.0 * base, C_bwd = 2.0 * base
  - KAN: static_macs include le basi B-spline ricorsive (Cox-de Boor cubico ~31.5 MAC/feature)
  - LTC: static_macs include la cella ncps con S = ode_unfolds = 6 sottopassi ODE
  - SNN (BPTT): C_fwd = T * F_analogico + 0.5 * r * T * F_spiking (primo layer a costo pieno); C_bwd = 2.0 * T * base
  - SNN (STDP): Forward sparsi guidati da r_pre; C_bwd = 0.5 * (r_pre + r_post) * T * base (LTP e LTD associativi, NO BPTT)
  - Forward-Forward MLP: C_fwd = 2.0 * sum(M_0..M_k), C_bwd = 2.0 * M_k (forward sui layer congelati + update del solo layer k)
  - Forward-Forward ResNet: C_fwd = 2.0 * sum(B_0..B_k), C_bwd = 4.0 * B_k (backward intra-blocco sul solo blocco k attivo)
  - CHL: C_fwd = T_free * (2*base - M_0) + T_clamped * (2*base - M_0 - M_last), C_bwd = 2.0 * base (2 correlazioni)
"""

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


class BudgetExceededException(Exception):
    """Eccezione sollevata quando il budget computazionale massimo è stato superato."""
    pass


def _conv2d_macs(mod: nn.Conv2d, out: Any) -> int:
    out_tensor = out[0] if isinstance(out, tuple) else out
    kh, kw = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
    cin = mod.in_channels // mod.groups
    # shape[-2:] funziona sia per tensori 4D [B, C, H, W] che 5D [T, B, C, H, W] (SpikingJelly step_mode='m')
    return int(cin * kh * kw * mod.out_channels * out_tensor.shape[-2] * out_tensor.shape[-1])


def _kan_linear_macs(mod: nn.Module) -> int:
    in_f = getattr(mod, "in_features", 0)
    out_f = getattr(mod, "out_features", 0)
    grid_size = getattr(mod, "grid_size", 5)
    spline_order = getattr(mod, "spline_order", 3)
    num_bases = grid_size + spline_order
    # Proiezione combinata basi spline + base SiLU: in_f * out_f * (1 + num_bases)
    proj_macs = in_f * out_f * (1 + num_bases)
    # Valutazione ricorsiva basi B-spline (algoritmo di Cox-de Boor cubico: ~31.5 MAC per input feature)
    spline_eval_macs = int(31.5 * in_f)
    return int(proj_macs + spline_eval_macs)


def _ltc_module_macs(mod: nn.Module, inp: Any, default_num_steps: int) -> int:
    rnn_cell = getattr(mod, "rnn_cell", getattr(mod, "cell", None))
    units = getattr(mod, "state_size", getattr(mod, "units", None))
    if hasattr(units, "units"):
        units = units.units
    elif not isinstance(units, int):
        units = getattr(rnn_cell, "state_size", 64)
    in_features = getattr(mod, "input_size", units)

    inp_tensor = inp[0] if isinstance(inp, tuple) and inp else inp
    if isinstance(inp_tensor, torch.Tensor) and inp_tensor.ndim == 3:
        batch_first = getattr(mod, "batch_first", True)
        num_steps = inp_tensor.shape[1] if batch_first else inp_tensor.shape[0]
    else:
        num_steps = default_num_steps

    # ncps.torch.LTCCell espone `_ode_unfolds` (valore predefinito = 6)
    ode_unfolds = getattr(
        mod,
        "ode_unfolds",
        getattr(rnn_cell, "_ode_unfolds", getattr(rnn_cell, "ode_unfolds", 6)),
    )
    c_sens = 2.5 * in_features * units
    c_rec = 2.5 * (units ** 2)
    c_state = 3.0 * units
    step_macs = int(c_sens + ode_unfolds * (c_rec + c_state))
    return int(step_macs * num_steps)


def _prepare_dummy_input(
    model: nn.Module,
    sample_input: Optional[torch.Tensor],
    input_dim: Optional[Any],
    dev: torch.device,
) -> torch.Tensor:
    dummy = sample_input
    if dummy is None:
        inp_d = input_dim if input_dim is not None else getattr(model, "input_dim", None)
        if inp_d is not None:
            shape = (1, int(inp_d)) if isinstance(inp_d, (int, float)) else (1, *inp_d)
            dummy = torch.zeros(*shape, device=dev)
        else:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    in_c = getattr(model, "in_channels", None) or m.in_channels
                    dummy = torch.zeros(1, in_c, 32, 32, device=dev)
                    break
                if isinstance(m, nn.Linear):
                    dummy = torch.zeros(1, m.in_features, device=dev)
                    break
            if dummy is None:
                dummy = torch.zeros(1, 784, device=dev)

    return dummy[:1] if dummy.shape[0] != 1 else dummy


def _collect_fallback_macs(model: nn.Module) -> Dict[str, int]:
    fallback: Dict[str, int] = {}
    for pname, p in model.named_parameters():
        if p.ndim == 2:
            fallback[pname] = int(p.numel())
        elif p.ndim == 4:
            fallback[pname] = int(p.numel() * 28 * 28)
    return fallback


def compute_model_static_macs(
    model: nn.Module,
    sample_input: Optional[torch.Tensor] = None,
    input_dim: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> int:
    """
    Calcola staticamente il numero teorico di MAC per un singolo campione nel forward pass.
    Supporta Conv2d, Linear, KANLinear e celle ricorrenti LTC con solver ODE.
    Popola sul modello `_macs_by_module`, `_first_layer_static_macs` e `_last_layer_static_macs`.
    """
    macs_dict: Dict[str, int] = {}
    hooks = []
    default_steps = getattr(model, "num_steps", 16)

    def make_hook(name: str):
        def hook(mod, inp, out):
            mod_class = mod.__class__.__name__
            if isinstance(mod, nn.Conv2d):
                macs_dict[name] = _conv2d_macs(mod, out)
            elif "KANLinear" in mod_class:
                macs_dict[name] = _kan_linear_macs(mod)
            elif mod_class == "LTC":
                macs_dict[name] = _ltc_module_macs(mod, inp, default_steps)
            elif isinstance(mod, nn.Linear) or (hasattr(mod, "in_features") and hasattr(mod, "out_features")):
                macs_dict[name] = int(getattr(mod, "in_features", 0) * getattr(mod, "out_features", 0))
        return hook

    for name, m in model.named_modules():
        cls_name = m.__class__.__name__
        if (
            isinstance(m, (nn.Linear, nn.Conv2d))
            or "KANLinear" in cls_name
            or cls_name == "LTC"
        ):
            hooks.append(m.register_forward_hook(make_hook(name)))

    dev = device or (next(model.parameters()).device if list(model.parameters()) else torch.device("cpu"))
    was_training = model.training
    model.eval()

    dummy = _prepare_dummy_input(model, sample_input, input_dim, dev)

    # Non alterare `model.num_steps`: reti come LiquidNetwork/CNNLiquidNetwork eseguono view con T fisso.
    try:
        with torch.no_grad():
            model(dummy.to(dev))
    except Exception as exc:
        if macs_dict:
            print(f"[BudgetTracker] Avviso: forward statico interrotto parzialmente ({exc}).")
    finally:
        try:
            from spikingjelly.activation_based import functional as sj_functional
            sj_functional.reset_net(model)
        except Exception:
            pass
        if was_training:
            model.train()
        for h in hooks:
            h.remove()

    if not macs_dict:
        macs_dict = _collect_fallback_macs(model)

    total_macs = sum(macs_dict.values())
    first_macs = next(iter(macs_dict.values()), 0) if macs_dict else 0
    last_macs = next(reversed(macs_dict.values()), 0) if macs_dict else 0
    setattr(model, "_macs_by_module", dict(macs_dict))
    setattr(model, "_first_layer_static_macs", first_macs)
    setattr(model, "_last_layer_static_macs", last_macs)

    return max(1, total_macs)


class BudgetTracker:
    """
    Traccia ed impone un budget computazionale fisso durante l'addestramento.
    Conteggio puramente aritmetico in MAC teorici (Multiply-Accumulate).
    """

    ALPHA_SOP = 0.50

    def __init__(
        self,
        target_macs: Optional[Union[float, int, str]] = None,
        enabled: bool = True,
        paradigm: str = "ann",
        tag: str = "benchmark",
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ):
        try:
            self.target_macs: Optional[float] = float(target_macs) if target_macs is not None else None
        except (ValueError, TypeError):
            self.target_macs = None

        self.enabled = bool(enabled and self.target_macs and self.target_macs > 0)
        self.paradigm = (paradigm or "ann").lower()
        self.tag = tag
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "results"

        # Metriche statiche modello (popolate da attach_model)
        self.static_fwd_macs: int = 0
        self.first_layer_static_macs: int = 0
        self.last_layer_static_macs: int = 0
        self.module_macs: Dict[str, int] = {}
        self.num_params: int = 0
        self.num_neurons: int = 0
        self.input_features: int = 784

        # Accumulatori
        self.cum_macs: float = 0.0
        self.cum_fwd_macs: float = 0.0
        self.cum_bwd_macs: float = 0.0
        self.samples_seen: int = 0
        self.exhausted: bool = False
        self.stop_epoch: Optional[int] = None

    def attach_model(
        self,
        model: nn.Module,
        sample_input: Optional[torch.Tensor] = None,
        input_dim: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Collega il modello e calcola parametri e MAC statici per singolo campione."""
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.num_params = params or sum(p.numel() for p in model.parameters())
        self.static_fwd_macs = compute_model_static_macs(model, sample_input, input_dim, device)
        self.first_layer_static_macs = getattr(model, "_first_layer_static_macs", 0)
        self.last_layer_static_macs = getattr(model, "_last_layer_static_macs", 0)
        self.module_macs = dict(getattr(model, "_macs_by_module", {}))
        self.num_neurons = max(1, sum(m.out_features for m in model.modules() if isinstance(m, nn.Linear)))

        self.input_features = 784
        if sample_input is not None:
            self.input_features = sample_input.flatten(1).shape[1] if sample_input.ndim > 1 else sample_input.numel()
        elif input_dim is not None:
            self.input_features = int(input_dim) if isinstance(input_dim, (int, float)) else int(torch.prod(torch.tensor(input_dim)))

        status = f"ATTIVO (Target: {self.target_macs:.2e} MAC)" if self.enabled else "PASSIVO (Solo monitoraggio)"
        print(f"[BudgetTracker] Inizializzato per paradigma '{self.paradigm}' [{status}]")
        print(f"[BudgetTracker] Parametri: {self.num_params:,} | MAC statici/campione forward: {self.static_fwd_macs:,}")

    def _sum_prefix_macs(self, prefix: str) -> int:
        return sum(
            v for k, v in self.module_macs.items()
            if k == prefix or k.startswith(prefix + ".")
        )

    def _resolve_ff_stage_macs(self, stage_prefix: str, stage_idx: Optional[int]) -> Tuple[int, int]:
        """Restituisce `(active_macs, upto_macs)` per il blocco/layer FF corrente."""
        if stage_idx is not None and self.module_macs:
            active_macs = self._sum_prefix_macs(f"{stage_prefix}.{stage_idx}")
            upto_macs = sum(self._sum_prefix_macs(f"{stage_prefix}.{i}") for i in range(stage_idx + 1))
            if active_macs > 0 and upto_macs > 0:
                return active_macs, upto_macs
        return self.static_fwd_macs, self.static_fwd_macs

    def compute_step_macs(
        self,
        batch_size: int,
        firing_rate: float = 0.0,
        num_steps_snn: int = 1,
        chl_steps: Optional[int] = None,
        chl_free_steps: Optional[int] = None,
        chl_clamped_steps: Optional[int] = None,
        is_ff_step: bool = False,
        ff_stage_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Calcolo del budget sinaptico teorico stimato in MAC equivalenti per un batch.
        """
        base_fwd = self.static_fwd_macs * batch_size
        p = self.paradigm

        if "stdp" in p:
            r_post = firing_rate
            r_pre = kwargs.get("firing_rate_pre", 0.20)
            c_fwd = 0.50 * r_pre * num_steps_snn * base_fwd
            c_bwd = 0.50 * (r_pre + r_post) * num_steps_snn * base_fwd
        elif "snn" in p:
            f_analog = self.first_layer_static_macs * batch_size
            f_spiking = max(0, base_fwd - f_analog)
            if f_analog > 0 and f_spiking > 0:
                c_fwd = (num_steps_snn * f_analog) + (0.50 * firing_rate * num_steps_snn * f_spiking)
            else:
                c_fwd = 0.50 * firing_rate * num_steps_snn * base_fwd
            c_bwd = 2.0 * base_fwd * num_steps_snn
        elif "ff_resnet" in p:
            active_macs, upto_macs = self._resolve_ff_stage_macs("blocks", ff_stage_idx)
            c_fwd = float(2.0 * upto_macs * batch_size)
            c_bwd = float(4.0 * active_macs * batch_size)
        elif "ff" in p or is_ff_step:
            active_macs, upto_macs = self._resolve_ff_stage_macs("layers", ff_stage_idx)
            c_fwd = float(2.0 * upto_macs * batch_size)
            c_bwd = float(2.0 * active_macs * batch_size)
        elif "chl" in p:
            m_first = self.first_layer_static_macs
            m_last = self.last_layer_static_macs
            free_step_macs = max(0, 2 * self.static_fwd_macs - m_first)
            clamped_step_macs = max(0, 2 * self.static_fwd_macs - m_first - m_last)
            if chl_free_steps is not None or chl_clamped_steps is not None:
                f_steps = chl_free_steps or 0
                cl_steps = chl_clamped_steps or 0
                c_fwd = float((f_steps * free_step_macs + cl_steps * clamped_step_macs) * batch_size)
            else:
                steps_rec = chl_steps if chl_steps is not None else 50
                half = steps_rec / 2.0
                c_fwd = float(half * (free_step_macs + clamped_step_macs) * batch_size)
            c_bwd = float(2.0 * base_fwd)
        else:
            c_fwd = float(base_fwd)
            c_bwd = 2.0 * c_fwd

        return {"fwd": c_fwd, "bwd": c_bwd, "total": c_fwd + c_bwd}

    def step(
        self,
        batch_size: int,
        firing_rate: float = 0.0,
        num_steps_snn: int = 1,
        chl_steps: Optional[int] = None,
        chl_free_steps: Optional[int] = None,
        chl_clamped_steps: Optional[int] = None,
        is_ff_step: bool = False,
        ff_stage_idx: Optional[int] = None,
        epoch: int = 1,
        **kwargs: Any,
    ) -> bool:
        """Registra un batch. Restituisce True se il budget è stato superato."""
        costs = self.compute_step_macs(
            batch_size=batch_size,
            firing_rate=firing_rate,
            num_steps_snn=num_steps_snn,
            chl_steps=chl_steps,
            chl_free_steps=chl_free_steps,
            chl_clamped_steps=chl_clamped_steps,
            is_ff_step=is_ff_step,
            ff_stage_idx=ff_stage_idx,
            **kwargs,
        )

        self.cum_fwd_macs += costs["fwd"]
        self.cum_bwd_macs += costs["bwd"]
        self.cum_macs += costs["total"]
        self.samples_seen += batch_size

        if self.enabled and self.target_macs and self.cum_macs >= self.target_macs:
            if not self.exhausted:
                self.exhausted = True
                self.stop_epoch = epoch
                pct = (self.cum_macs / self.target_macs) * 100.0
                print(f"\n{'!' * 70}")
                print("[BudgetTracker] 🛑 BUDGET COMPUTAZIONALE RAGGIUNTO!")
                print(f"[BudgetTracker] MAC teorici stimati: {self.cum_macs:.3e} / {self.target_macs:.3e} ({pct:.1f}%)")
                print(f"[BudgetTracker] Campioni processati: {self.samples_seen:,} (Epoca: {epoch})")
                print(f"{'!' * 70}\n")
            return True

        return False

    def __bool__(self) -> bool:
        return self.enabled

    @property
    def is_exhausted(self) -> bool:
        return self.exhausted

    def print_summary(self) -> None:
        """Stampa un riepilogo scientifico del costo di addestramento."""
        total = max(1.0, self.cum_macs)
        pct_fwd = (self.cum_fwd_macs / total) * 100.0
        pct_bwd = (self.cum_bwd_macs / total) * 100.0

        status = "RAGGIUNTO (Stop)" if self.exhausted else ("IN CORSO" if self.enabled else "DISATTIVATO")
        print("\n" + "=" * 65)
        print("    RIEPILOGO BUDGET COMPUTAZIONALE (Budget Sinaptico Teorico MAC_eq)")
        print("=" * 65)
        print(f"  Paradigma di calcolo:       {self.paradigm.upper()}")
        print(f"  Stato del vincolo:          {status}")
        if self.target_macs:
            print(f"  Budget prefissato:          {self.target_macs:.3e} MAC_eq")
        print(f"  MAC_eq totali stimati:      {self.cum_macs:.3e} MAC_eq")
        print(f"  Campioni di training visti: {self.samples_seen:,}")
        print("-" * 65)
        print("  Ripartizione del Costo di Addestramento:")
        print(f"    - Forward pass (C_fwd):   {self.cum_fwd_macs:.3e} MAC_eq  ({pct_fwd:5.1f}%)")
        print(f"    - Backward/Update (C_bwd):{self.cum_bwd_macs:.3e} MAC_eq  ({pct_bwd:5.1f}%)")
        print("=" * 65 + "\n")

    def export_csv(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Esporta il riepilogo finale del budget in CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = Path(filepath) if filepath else self.output_dir / f"{self.tag}_budget_summary.csv"

        fieldnames = [
            "tag", "paradigm", "target_macs", "cum_macs", "cum_fwd_macs",
            "cum_bwd_macs", "samples_seen", "budget_exhausted",
        ]
        row = {
            "tag": self.tag,
            "paradigm": self.paradigm,
            "target_macs": self.target_macs,
            "cum_macs": round(self.cum_macs, 2),
            "cum_fwd_macs": round(self.cum_fwd_macs, 2),
            "cum_bwd_macs": round(self.cum_bwd_macs, 2),
            "samples_seen": self.samples_seen,
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
    """Crea o recupera il BudgetTracker da args/profiler e collega il modello se fornito."""
    if profiler is not None and getattr(profiler, "budget_tracker", None) is not None:
        tracker = profiler.budget_tracker
        if model is not None and tracker.static_fwd_macs == 0:
            tracker.attach_model(model, sample_input=sample_input, device=device)
        return tracker

    b_cfg = getattr(args, "budget", None)
    enabled = getattr(b_cfg, "enabled", False) if b_cfg else False
    max_macs = getattr(b_cfg, "max_macs", None) if b_cfg else None
    if not enabled or not max_macs or float(max_macs) <= 0:
        return None

    paradigm_name = paradigm or getattr(args, "paradigm", "ann")
    tag = f"{paradigm_name}"
    if hasattr(args, "dataset") and hasattr(args.dataset, "name"):
        tag += f"_{args.dataset.name}"

    tracker = BudgetTracker(
        target_macs=max_macs,
        enabled=enabled,
        paradigm=paradigm_name,
        tag=tag,
        output_dir=getattr(args, "results_dir", None),
    )

    if model is not None:
        tracker.attach_model(model, sample_input=sample_input, device=device)
    if profiler is not None:
        profiler.budget_tracker = tracker

    return tracker
