import torch
def get_device(device=None):
    """Restituisce il dispositivo PyTorch appropriato."""
    if isinstance(device, torch.device):
        return device

    if device and device != "auto":
        return torch.device(device)

    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )