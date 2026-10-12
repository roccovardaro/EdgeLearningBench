"""Modulo per la configurazione centralizzata di optimizer, scheduler e criterion."""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR


# ── Loss per Spiking Neural Networks ──────────────────────────────────────────

class SNNLoss(nn.Module):
    """Loss per SNN: Cross-Entropy sul potenziale di membrana medio + spike rate."""

    def __init__(self, mem_weight=1.0, spike_weight=1.0, label_smoothing=0.0):
        super().__init__()
        self.mem_weight = mem_weight
        self.spike_weight = spike_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, spk_out, mem_out, targets):
        mem_loss = self.ce(mem_out.mean(dim=0), targets)
        spike_loss = self.ce(spk_out.sum(dim=0), targets)
        return self.mem_weight * mem_loss + self.spike_weight * spike_loss


SNNCNNLoss = SNNLoss


# ── Builder Unificato ────────────────────────────────────────────────────────

def get_training_components(model, args, model_type="default"):
    """Restituisce (optimizer, scheduler, criterion) per il modello specificato.

    Args:
        model: Il modello PyTorch (nn.Module).
        args: Namespace degli argomenti (args del paradigma o args.train).
        model_type: Nome del modello ('resnet', 'mlp', 'kan', 'cnn_kan',
                    'ltc', 'cnn_ltc', 'snn_mlp', 'snn_cnn').
    """
    train = getattr(args, "train", args)
    epochs = train.epochs
    lr = train.lr
    weight_decay = getattr(train, "weight_decay", 1e-4)

    # 1. Optimizer: AdamW comune a tutti i modelli
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 2. Scheduler: Cosine Annealing comune a tutti i modelli
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Criterion: CrossEntropy (con label smoothing se previsto) o SNNLoss
    m_type = model_type.lower()
    if m_type in ("snn_mlp", "snn_cnn", "snn"):
        mem_weight = getattr(train, "mem_weight", 1.0)
        spike_weight = getattr(train, "spike_weight", 1.0)
        criterion = SNNLoss(mem_weight=mem_weight, spike_weight=spike_weight)
    elif m_type in ("resnet", "mlp", "kan"):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()

    return optimizer, scheduler, criterion
