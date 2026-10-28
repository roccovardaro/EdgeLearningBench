"""Modulo per la configurazione centralizzata di optimizer e criterion."""

import torch
import torch.nn as nn


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


class SNNRegressionLoss(nn.Module):
    """Loss per SNN in regressione: MSE sul potenziale medio + opzionale penalità su spike rate."""

    def __init__(self, spike_weight=0.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.spike_weight = spike_weight

    def forward(self, spk_out, mem_out, targets):
        pred = mem_out.mean(dim=0).view(-1, 1)
        targets = targets.view(-1, 1)
        loss = self.mse(pred, targets)
        if self.spike_weight > 0 and spk_out is not None:
            loss = loss + self.spike_weight * spk_out.mean()
        return loss


SNNCNNLoss = SNNLoss


# ── Builder Unificato ────────────────────────────────────────────────────────

def get_training_components(model, args, model_type="default"):
    """Restituisce (optimizer, criterion) per il modello specificato.

    Args:
        model: Il modello PyTorch (nn.Module).
        args: Namespace degli argomenti (args del paradigma o args.train).
        model_type: Nome del modello ('resnet', 'mlp', 'kan', 'cnn_kan',
                    'ltc', 'cnn_ltc', 'snn_mlp', 'snn_cnn').
    """
    train = getattr(args, "train", args)
    lr = train.lr
    weight_decay = getattr(train, "weight_decay", 1e-4)

    # 1. Optimizer: AdamW comune a tutti i modelli (parametri con gradiente abilitato)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # 2. Criterion: MSELoss per regressione (num_classes == 1), altrimenti CrossEntropy o SNNLoss
    num_classes = getattr(getattr(args, "model", None), "num_classes", None)
    if num_classes is None:
        num_classes = getattr(args, "num_classes", None)

    m_type = model_type.lower()
    if num_classes == 1:
        if m_type in ("snn_mlp", "snn_cnn", "snn_resnet", "snn"):
            spike_weight = getattr(train, "spike_weight", 0.0)
            criterion = SNNRegressionLoss(spike_weight=spike_weight)
        else:
            criterion = nn.MSELoss()
    elif m_type in ("snn_mlp", "snn_cnn", "snn_resnet", "snn"):
        mem_weight = getattr(train, "mem_weight", 1.0)
        spike_weight = getattr(train, "spike_weight", 1.0)
        criterion = SNNLoss(mem_weight=mem_weight, spike_weight=spike_weight)
    elif m_type in ("resnet", "mlp", "cnn", "kan", "direct_kan"):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()

    return optimizer, criterion
