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


@torch.no_grad()
def regression_metrics(model, loader, device=None):
    """Calcola le metriche di regressione (MSE, RMSE, MAE, R^2) su un DataLoader.

    Args:
        model: Il modello PyTorch (nn.Module).
        loader: DataLoader di training o test.
        device: Dispositivo su cui eseguire l'inferenza (se None usa get_device).

    Returns:
        dict: Dizionario contenente {'mse': float, 'rmse': float, 'mae': float, 'r2': float}.
    """
    device = get_device(device)
    was_training = model.training
    model.eval()

    all_preds, all_targets = [], []
    try:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device=device, dtype=torch.float32).view(-1, 1)

            out = model(inputs)
            if isinstance(out, tuple):
                # SNN restituisce (spk_recordings, mem_recordings)
                mem_out = out[1] if len(out) > 1 else out[0]
                if mem_out.ndim == 3:  # [num_steps, B, D]
                    preds = mem_out.mean(dim=0).view(-1, 1)
                else:
                    preds = mem_out.view(-1, 1)
            else:
                preds = out.view(-1, 1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

            sj_functional.reset_net(model)

        if not all_preds:
            raise ValueError("Il DataLoader di valutazione è vuoto.")

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        mse = torch.mean((preds - targets) ** 2).item()
        rmse = float(torch.sqrt(torch.tensor(mse)).item())
        mae = torch.mean(torch.abs(preds - targets)).item()

        ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
        ss_res = torch.sum((targets - preds) ** 2).item()
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
    finally:
        model.train(was_training)
