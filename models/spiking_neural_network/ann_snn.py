from pathlib import Path
import torch
import torch.nn as nn
from device import get_device
from paths import CHECKPOINT_DIR
from models.baseline.resnet import resnet

try:
    from spikingjelly.activation_based import ann2snn, functional
except ImportError:
    ann2snn = None
    functional = None


def load_ann_from_checkpoint(checkpoint_path, device):
    """Carica il checkpoint salvato da resnet_run e ricostruisce la ResNet."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        ckpt_path = CHECKPOINT_DIR / checkpoint_path
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ckpt.get("model_config", {})
    model = resnet(
        num_classes=cfg.get("num_classes", 10),
        in_channels=cfg.get("in_channels", 3),
        layers=cfg.get("layers", [9, 9, 9]),
    ).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[ANN] Pesi caricati con successo da {ckpt_path}")
    return model


def prepare_ann_for_snn(model):
    """Disattiva inplace=True nei layer ReLU per compatibilità con SpikingJelly."""
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model


def convert_to_snn(ann_model, train_loader, device, mode="99.9%"):
    """Converte la ResNet in SNN calibrando le soglie sul train_loader."""
    if ann2snn is None:
        raise ImportError("SpikingJelly non è installata. Esegui: pip install spikingjelly")

    ann_model = prepare_ann_for_snn(ann_model)
    ann_model.eval()

    # Calibrazione veloce sui primi 10 batch (1280 immagini)
    calib_batches = [b for i, b in enumerate(train_loader) if i < 10]

    print(f"Conversione ANN -> SNN in corso (mode={mode})...")
    recipe = ann2snn.RateCodingRecipe(dataloader=calib_batches, mode=mode)
    converter = ann2snn.Converter(recipe=recipe, device=device)
    snn_model = converter.convert(ann_model).to(device)
    snn_model.eval()
    return snn_model


@torch.no_grad()
def evaluate_snn(snn_model, test_loader, device, T=16):
    """Valuta la SNN sui dati di test accumulando gli spike per T timestep."""
    snn_model.eval()
    correct, total = 0, 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        out_spikes_sum = 0.0
        for _ in range(T):
            out_spikes_sum += snn_model(images)

        correct += (out_spikes_sum.argmax(1) == labels).sum().item()
        total += labels.size(0)

        if functional is not None:
            functional.reset_net(snn_model)

    acc = correct / total
    print(f"SNN Test Accuracy (T={T}): {acc:.4f} (error: {1 - acc:.4f})")
    return acc


# ── Entry point per la conversione ───────────────────────────────────────────

def snn_run(checkpoint_path, train_loader, test_loader, device=None, T=16, mode="99.9%"):
    """Funzione principale per la conversione da checkpoint."""
    device = get_device(device)
    if device.type == "mps":
        print("[SpikingJelly] I neuroni SpikingJelly supportano solo CPU e NVIDIA CUDA (non Apple Silicon MPS).")
        print("[SpikingJelly] L'esecuzione viene commutata automaticamente su CPU.")
        device = torch.device("cpu")

    # 1. Carica la ResNet pre-addestrata
    ann_model = load_ann_from_checkpoint(checkpoint_path, device)

    # 2. Converti in SNN
    snn_model = convert_to_snn(ann_model, train_loader, device, mode=mode)

    # 3. Valuta la SNN
    snn_acc = evaluate_snn(snn_model, test_loader, device, T=T)

    return snn_model, snn_acc


def ann_snn_run(args, train_loader, test_loader, profiler=None, device=None):
    """Entry point compatibile con main.py."""
    device = get_device(device)
    if device.type == "mps":
        print("[SpikingJelly] I neuroni SpikingJelly supportano solo CPU e NVIDIA CUDA (non Apple Silicon MPS).")
        print("[SpikingJelly] L'esecuzione viene commutata automaticamente su CPU.")
        device = torch.device("cpu")

    checkpoint_name = getattr(args.model, "baseline_checkpoint", "resnet.pt")
    T = getattr(args.model, "num_steps", 16)
    mode = getattr(args.model, "conversion_mode", "99.9%")

    snn_model, snn_acc = snn_run(
        checkpoint_name,
        train_loader,
        test_loader,
        device=device,
        T=T,
        mode=mode,
    )

    if profiler:
        profiler.log_eval(0.0, snn_acc)

    # Salva il checkpoint del modello SNN convertito
    checkpoint_out = getattr(args.model, "checkpoint_name", "ann_snn.pt")
    checkpoint_path = CHECKPOINT_DIR / checkpoint_out
    torch.save(snn_model.state_dict(), checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return snn_model