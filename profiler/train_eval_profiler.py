"""
TrainEvalProfiler: Profiler leggero per salvare loss, accuracy e tempo per epoca di train ed evaluation.
"""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrainEvalProfiler:
    """Registra metriche di training e valutazione ed esporta in CSV (epoch, loss, accuracy, time)."""

    def __init__(self, tag: str = "benchmark", output_dir: Optional[Path] = None):
        self.tag = tag
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "results"
        self.train_records: List[Dict[str, Any]] = []
        self.eval_records: List[Dict[str, Any]] = []
        self._start_time = time.time()

    def log_epoch(
        self,
        epoch: int,
        loss: float,
        acc: Optional[float] = None,
        time_sec: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Registra metriche di un'epoca di train: epoch, loss, accuracy, time."""
        now = time.time()
        elapsed = time_sec if time_sec is not None else (now - self._start_time)
        self._start_time = now

        self.train_records.append({
            "epoch": epoch,
            "loss": round(float(loss), 4),
            "accuracy": round(float(acc), 4) if acc is not None else None,
            "time": round(float(elapsed), 4),
        })

    def log_eval(
        self,
        epoch: Any = 1,
        loss: Optional[float] = None,
        acc: Optional[float] = None,
        time_sec: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Registra metriche di valutazione: epoch, loss, accuracy, time."""
        # Supporta sia log_eval(epoch, loss, acc) che la chiamata classica log_eval(train_acc, test_acc)
        if isinstance(epoch, float) and loss is not None:
            acc = float(loss)  # test_acc
            loss = None
            epoch = len(self.train_records) or 1

        self.eval_records.append({
            "epoch": epoch,
            "loss": round(float(loss), 4) if loss is not None else None,
            "accuracy": round(float(acc), 4) if acc is not None else None,
            "time": round(float(time_sec), 4) if time_sec is not None else round(time.time() - self._start_time, 4),
        })

    def export_csv(self) -> Dict[str, Optional[Path]]:
        """Esporta due file CSV: uno per il train e uno per l'eval."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fields = ["epoch", "loss", "accuracy", "time"]
        paths: Dict[str, Optional[Path]] = {"train": None, "eval": None}

        for mode, records in [("train", self.train_records), ("eval", self.eval_records)]:
            if records:
                filepath = self.output_dir / f"{self.tag}_{mode}_{ts}.csv"
                with open(filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(records)
                print(f"[TrainEvalProfiler] Dati {mode} esportati in: {filepath}")
                paths[mode] = filepath

        return paths

    save = export_csv


