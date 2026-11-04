"""Modulo per il calcolo delle metriche di valutazione delle performance."""

import torch

from device import get_device


from spikingjelly.activation_based import functional as sj_functional


@torch.no_grad()
def accuracy(model, loader, device=None):
    """Calcola l'accuratezza su un DataLoader per qualsiasi modello supportato.
    Args:
        model: Il modello PyTorch (nn.Module).
        loader: DataLoader di training o test.
        device: Dispositivo su cui eseguire l'inferenza (se None usa get_device).

    Returns:
        float: Frazione di predizioni corrette (tra 0.0 e 1.0).
    """
    device = get_device(device)
    was_training = model.training
    model.eval()
    correct, total = 0, 0
    try:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device=device, dtype=torch.long)

            if hasattr(model, "predict") and callable(getattr(model, "predict")):
                preds = model.predict(images)  # Modelli forward forward
            else:
                out = model(images)
                if isinstance(out, tuple):  # Modelli Spiking neural network snn_mlp, snn_cnn
                    # SNN restituisce (spk_recordings, mem_recordings)
                    mem_out = out[1] if len(out) > 1 else out[0]
                    preds = mem_out.mean(dim=0).argmax(dim=1)
                else:
                    preds = out.argmax(dim=1)  # Modelli Classici mlp, cnn_kan, resnet, kan, ltc

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            sj_functional.reset_net(model)

        if total == 0:
            raise ValueError("Il DataLoader di valutazione è vuoto.")
        return correct / total
    finally:
        model.train(was_training)

