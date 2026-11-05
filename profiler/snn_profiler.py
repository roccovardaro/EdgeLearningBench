"""
SNNProfiler: Profiler per Spiking Neural Networks con stima energetica neuromorfica.

Monitora:
  - Firing rate e sparsity degli spike su tutti i neuroni LIF/IF
  - Synaptic Operations (SOPs) stimate
  - Energia neuromorfica stimata (E_SNN) basata su tecnologia CMOS a 45nm (Horowitz 2014)
  - Confronto con l'ANN equivalente (E_ANN e fattore di risparmio energetico)
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from profiler.budget_tracker import compute_model_static_macs as compute_model_macs


class SNNProfiler:
    """Profiler SNN: traccia spike, sparsity ed energia neuromorfica."""

    # Costanti di riferimento standard (CMOS 45nm, Mark Horowitz 2014)
    E_AC = 0.9e-12   # 0.9 pJ per Accumulo (Synaptic Operation neuromorfica)
    E_MAC = 4.6e-12  # 4.6 pJ per Multiply-Accumulate (ANN convenzionale)

    def __init__(
        self,
        tag: str = "snn",
        output_dir: Optional[Union[str, Path]] = None,
        model: Optional[nn.Module] = None,
        num_steps: Optional[int] = None,
    ):
        self.tag = tag
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "results"
        self.records: List[Dict[str, Any]] = []

        # Hook per intercettare gli spike da tutti i neuroni LIF/IF
        self._hooks = []
        self._has_hooks = False

        # Accumulatori per l'epoca corrente
        self._epoch_spikes = 0
        self._epoch_elements = 0

        self.num_steps = num_steps or (getattr(model, "num_steps", 1) if model else 1)
        self.total_macs = compute_model_macs(model) if model is not None else 0 #Calcola una sola volta i MAC statici totali del modello

        if model is not None:
            self._attach_hooks(model)

    def _attach_hooks(self, model: nn.Module):
        for m in model.modules():
            if m.__class__.__name__ in (
                "Leaky", "Synaptic", "Alpha", "IFNode", "LIFNode", "ParametricLIFNode",
                "AdaptiveLIFNode", "_AdaptiveLIFNode", "SimpleLIFNode", "SimpleIFNode",
            ):
                self._has_hooks = True
                self._hooks.append(m.register_forward_hook(self._hook_fn))

    def _hook_fn(self, module, inp, out):
        spk = out[0] if isinstance(out, tuple) else out #gestice caso in cui il layer restituisca una tupla (spk, mem) come in snnTorch
        if isinstance(spk, torch.Tensor):
            self._epoch_spikes += spk.detach().sum().item() #Somma totale degli 1 (spikes) emessi
            self._epoch_elements += spk.numel() #Numero totale di elementi nei tensori(Batch * Neuroni per ogni step)

    def update(self, spk_out):
        """Accumula statistiche dal batch (usato se non sono attivi hook globali)."""
        if not self._has_hooks:
            self._epoch_spikes += spk_out.detach().sum().item()
            self._epoch_elements += spk_out.numel()

    @property
    def current_firing_rate(self) -> float:
        """Restituisce il firing rate globale medio accumulato fino all'istante corrente."""
        return (self._epoch_spikes / self._epoch_elements) if self._epoch_elements > 0 else 0.0

    def reset(self):
        """Azzera gli accumulatori di spike."""
        self._epoch_spikes = 0
        self._epoch_elements = 0

    def log_epoch(self, epoch: Any = 1, tag: str = "train", verbose: bool = False) -> Dict[str, Any]:
        """Chiude l'epoca o fase (train/test), registra e resetta gli accumulatori.
           Vengono calcolate le metriche chiave prima di azzerare i contatori per la fase successiva."""
        firing_rate = self.current_firing_rate
        sparsity = 1.0 - firing_rate

        # Stima neuromorfica per singolo campione
        sops_per_sample = int(firing_rate * self.num_steps * self.total_macs)
        e_snn_uj = sops_per_sample * self.E_AC * 1e6
        e_ann_uj = self.total_macs * self.E_MAC * 1e6
        energy_gain = (e_ann_uj / e_snn_uj) if e_snn_uj > 0 else 1.0

        record = {
            "tag": tag,
            "epoch": epoch,
            "sparsity": round(sparsity, 4),
            "firing_rate": round(firing_rate, 4),
            "total_spikes": int(self._epoch_spikes),
            "sops_per_sample": sops_per_sample,
            "snn_energy_uj": round(e_snn_uj, 3),
            "ann_energy_uj": round(e_ann_uj, 3),
            "energy_gain": round(energy_gain, 2),
        }
        self.records.append(record)

        if verbose:
            print(
                f"[SNNProfiler] [{tag.upper()}] Epoch {str(epoch):>3s} | "
                f"sparsity={sparsity*100:.1f}% | "
                f"firing_rate={firing_rate:.4f} | "
                f"E_SNN={e_snn_uj:.2f}µJ | "
                f"Gain={energy_gain:.2f}x"
            )

        # Reset per la prossima epoca o fase
        self.reset()

        return record

    def print_summary(self):
        """Stampa riepilogo finale comprensivo di stima energetica neuromorfica."""
        if not self.records:
            return

        # Mostra prioritariamente il record di test se presente, altrimenti l'ultimo di train
        test_records = [r for r in self.records if r.get("tag") == "test"]
        last = test_records[-1] if test_records else self.records[-1]
        phase = last.get("tag", "train").upper()

        print("\n" + "=" * 54)
        print(f"       SNN PROFILER — RIEPILOGO NEUROMORFICO ({phase})")
        print("=" * 54)
        print(f"  Fase (tag):                 {last.get('tag', 'train')}")
        print(f"  Sparsity finale:            {last['sparsity']*100:.1f}%")
        print(f"  Firing rate finale:         {last['firing_rate']:.4f}")
        print(f"  Spike totali:               {last['total_spikes']:,}")
        if self.total_macs > 0:
            print("-" * 54)
            print(f"  Operazioni Sinaptiche (SOP):{last['sops_per_sample']:,} / campione")
            print(f"  MACs ANN equivalenti:       {self.total_macs:,} / campione")
            print(f"  Energia SNN stimata:        {last['snn_energy_uj']:.3f} µJ / campione")
            print(f"  Energia ANN equivalente:    {last['ann_energy_uj']:.3f} µJ / campione")
            if last['energy_gain'] >= 1.0:
                print(f"  Efficienza Energetica:      {last['energy_gain']:.2f}x risparmio (SNN più efficiente)")
            else:
                print(f"  Efficienza Energetica:      {last['energy_gain']:.2f}x (ANN più efficiente con rate attuale)")
        print("=" * 54 + "\n")

    def export_csv(self) -> Optional[Path]:
        """Esporta le metriche per epoca in CSV."""
        if not self.records:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"{self.tag}_snn_metrics_{ts}.csv"

        fields = [
            "tag",
            "epoch",
            "sparsity",
            "firing_rate",
            "total_spikes",
            "sops_per_sample",
            "snn_energy_uj",
            "ann_energy_uj",
            "energy_gain",
        ]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.records)

        print(f"[SNNProfiler] CSV esportato in: {filepath}")
        return filepath

    def cleanup(self):
        """Rimuove gli hook registrati."""
        hooks = getattr(self, "_hooks", [])
        for h in hooks:
            h.remove()
        if hasattr(self, "_hooks"):
            self._hooks.clear()

    def __del__(self):
        self.cleanup()
