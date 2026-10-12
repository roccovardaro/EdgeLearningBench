"""
SNNProfiler: Profiler leggero per Spiking Neural Networks.

Raccoglie sparsity e firing rate direttamente dagli spike
già calcolati durante il training, senza forward pass aggiuntivi.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SNNProfiler:
    """Profiler SNN: accumula statistiche dagli spike del training."""

    def __init__(self, tag: str = "snn", output_dir: Optional[Path] = None):
        self.tag = tag
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "results"
        self.records: List[Dict[str, Any]] = []

        # Accumulatori per l'epoca corrente
        self._epoch_spikes = 0
        self._epoch_elements = 0

    # ── Raccolta batch per batch (costo ~zero) ───────────────────

    def update(self, spk_out):
        """Accumula statistiche da un singolo batch.

        Args:
            spk_out: Tensor [num_steps, B, num_classes] — spike di output.
        """
        self._epoch_spikes += spk_out.detach().sum().item()
        self._epoch_elements += spk_out.numel()

    def log_epoch(self, epoch: int, verbose: bool = False) -> Dict[str, Any]:
        """Chiude l'epoca corrente, registra e resetta gli accumulatori."""
        firing_rate = self._epoch_spikes / self._epoch_elements if self._epoch_elements > 0 else 0.0
        sparsity = 1.0 - firing_rate

        record = {
            "epoch": epoch,
            "sparsity": round(sparsity, 4),
            "firing_rate": round(firing_rate, 4),
            "total_spikes": int(self._epoch_spikes),
        }
        self.records.append(record)

        if verbose:
            print(
                f"[SNNProfiler] Epoch {epoch:3d} | "
                f"sparsity={sparsity:.4f} ({sparsity*100:.1f}%) | "
                f"firing_rate={firing_rate:.4f}"
            )

        # Reset per la prossima epoca
        self._epoch_spikes = 0
        self._epoch_elements = 0

        return record

    # ── Report finale ────────────────────────────────────────────

    def print_summary(self):
        """Stampa riepilogo finale."""
        if not self.records:
            return
        last = self.records[-1]
        print("\n" + "=" * 50)
        print("       SNN PROFILER — RIEPILOGO")
        print("=" * 50)
        print(f"  Sparsity finale:    {last['sparsity']:.4f}  ({last['sparsity']*100:.1f}%)")
        print(f"  Firing rate finale: {last['firing_rate']:.4f}")
        print(f"  Spike totali:       {last['total_spikes']:,}")
        print("=" * 50 + "\n")

    # ── Export CSV ────────────────────────────────────────────────

    def export_csv(self) -> Optional[Path]:
        """Esporta le metriche per epoca in CSV."""
        if not self.records:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"{self.tag}_snn_metrics_{ts}.csv"

        fields = ["epoch", "sparsity", "firing_rate", "total_spikes"]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.records)

        print(f"[SNNProfiler] CSV esportato in: {filepath}")
        return filepath
