"""
Modulo per il calcolo dei costi computazionali (in MAC) della conversione e calibrazione ANN -> SNN.
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from profiler.budget_tracker import compute_model_static_macs


def compute_conversion_costs(
    ann_model: nn.Module,
    train_loader: Optional[Any] = None,
    num_calib_samples: Optional[int] = None,
    sample_input: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Calcola in modo semplice i costi computazionali (in MAC) della conversione e calibrazione ANN -> SNN."""
    # 1. Numero di campioni di calibrazione (da parametro o dai primi 32 batch del train_loader)
    if num_calib_samples is not None:
        n_samples = int(num_calib_samples)
    elif train_loader is not None:
        batches = [b for i, b in enumerate(train_loader) if i < 32]
        n_samples = sum(len(b[0] if isinstance(b, (list, tuple)) else b) for b in batches)
        if sample_input is None and batches:
            sample_input = (batches[0][0] if isinstance(batches[0], (list, tuple)) else batches[0])[:1]
    else:
        n_samples = 32 * 128  # default: 4096 campioni

    # 2. MAC statici per singolo campione (forward pass ANN)
    macs_per_sample = compute_model_static_macs(ann_model, sample_input=sample_input)

    # 3. Costo totale di calibrazione (solo forward pass, nessun backward)
    total_macs = n_samples * macs_per_sample

    costs = {
        "macs_per_sample": macs_per_sample,
        "num_calib_samples": n_samples,
        "calibration_macs": total_macs,
        "calibration_gmac": round(total_macs / 1e9, 4),
        "backward_macs": 0,
    }

    if verbose:
        print("\n--- Costi Conversione/Calibrazione (in MAC) ---")
        print(f"  MAC statici/campione: {macs_per_sample:,} MAC")
        print(f"  Campioni calibrazione: {n_samples:,}")
        print(f"  Backward MAC:          0 MAC")
        print(f"  Totale Calibrazione:   {total_macs:,} MAC ({costs['calibration_gmac']} GMAC)\n")

    return costs
