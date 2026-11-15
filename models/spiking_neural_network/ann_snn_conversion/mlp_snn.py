"""
Modulo per la conversione di una rete MLP (Multi-Layer Perceptron) pre-addestrata
in una Spiking Neural Network (SNN) tramite SpikingJelly (ann2snn con Rate Coding).

Pipeline:
1. Caricamento pesi e architettura dal checkpoint della MLP addestrata (mlp.pt).
2. Preparazione modello per SNN (disattivazione inplace=True nei ReLU, eval mode).
3. Calibrazione soglie di sparo (V_th) sul training set tramite RateCodingRecipe.
4. Conversione strutturale via ann2snn.Converter (sostituzione ReLU -> IFNode).
5. Valutazione e accumulo degli spike su T timestep temporali.
6. Salvataggio del checkpoint della SNN convertita (mlp_snn.pt).
"""

from pathlib import Path
import torch
import torch.nn as nn

from device import get_device
from paths import CHECKPOINT_DIR, RESULTS_DIR
from profiler.snn_profiler import SNNProfiler
from models.baseline.mlp import build_mlp, MLPClassifier
from models.spiking_neural_network.ann_snn_conversion.costs import compute_conversion_costs

from spikingjelly.activation_based import ann2snn, functional


def load_mlp_from_checkpoint(checkpoint_path, device, fallback_cfg=None):
    """Carica il checkpoint salvato da mlp_run e ricostruisce la MLPClassifier."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        ckpt_path = CHECKPOINT_DIR / checkpoint_path
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint MLP non trovato: {checkpoint_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    fallback_cfg = fallback_cfg or {}
    cfg = ckpt.get("model_config", {})

    input_dim = cfg.get("input_dim", fallback_cfg.get("input_dim", 784))
    num_classes = cfg.get("num_classes", fallback_cfg.get("num_classes", 10))
    hidden_dims = cfg.get("hidden_dims", fallback_cfg.get("hidden_dims", (512, 256)))
    dropout = cfg.get("dropout", fallback_cfg.get("dropout", 0.2))

    model = build_mlp(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=tuple(hidden_dims),
        dropout=dropout,
    ).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[MLP-ANN] Pesi caricati con successo da {ckpt_path}")
    return model


def prepare_mlp_for_snn(model):
    """Disattiva inplace=True nei layer ReLU e imposta eval mode per SpikingJelly."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model


def convert_mlp_to_snn(ann_model, train_loader, device, mode="99.9%", num_calib_batches=32):
    """Converte la MLP addestrata in SNN calibrando le soglie di sparo sul train_loader."""
    ann_model = prepare_mlp_for_snn(ann_model)

    # Campioni per calibrazione delle soglie di attivazione
    calib_batches = [b for i, b in enumerate(train_loader) if i < num_calib_batches]

    print(f"[MLP-to-SNN] Conversione ANN -> SNN in corso (mode={mode}, {len(calib_batches)} batch di calibrazione)...")
    recipe = ann2snn.RateCodingRecipe(dataloader=calib_batches, mode=mode)
    converter = ann2snn.Converter(recipe=recipe, device=device)
    snn_model = converter.convert(ann_model).to(device)
    snn_model.eval()
    print("[MLP-to-SNN] Conversione completata con successo.")
    return snn_model


@torch.no_grad()
def evaluate_mlp_snn(snn_model, test_loader, device, T=32, snn_prof=None):
    """Valuta la SNN convertita accumulando le uscite su T timestep temporali."""
    snn_model.eval()
    correct, total = 0, 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        if snn_prof is not None:
            snn_prof.ensure_static_macs(images[:1])

        out_spikes_sum = 0.0
        for _ in range(T):
            out_spikes_sum += snn_model(images)

        correct += (out_spikes_sum.argmax(1) == labels).sum().item()
        total += labels.size(0)

        functional.reset_net(snn_model)

    acc = correct / total
    print(f"[MLP-SNN] Test Accuracy (T={T}): {acc:.4f} (error: {1 - acc:.4f})")
    return acc


def mlp_snn_run(args, train_loader, test_loader, profiler=None, device=None):
    """Entry point compatibile con main.py per la conversione MLP -> SNN."""
    device = get_device(device)
    if device.type == "mps":
        print("[SpikingJelly] I neuroni SpikingJelly supportano solo CPU e NVIDIA CUDA (non Apple Silicon MPS).")
        print("[SpikingJelly] L'esecuzione viene commutata automaticamente su CPU.")
        device = torch.device("cpu")

    checkpoint_name = getattr(args.model, "baseline_checkpoint", "mlp.pt")
    checkpoint_out = getattr(args.model, "checkpoint_name", "mlp_snn.pt")
    T = getattr(args.model, "num_steps", 32)
    mode = getattr(args.model, "conversion_mode", "99.9%")

    fallback_cfg = {
        "input_dim": getattr(args.model, "input_dim", 784),
        "num_classes": getattr(args.model, "num_classes", 10),
        "hidden_dims": getattr(args.model, "hidden_dims", (512, 256)),
        "dropout": getattr(args.model, "dropout", 0.2),
    }

    # 1. Carica la MLP pre-addestrata
    ann_model = load_mlp_from_checkpoint(checkpoint_name, device, fallback_cfg=fallback_cfg)

    # Calcola e stampa i costi computazionali di calibrazione e conversione (in MAC) per la tesi
    calib_costs = compute_conversion_costs(ann_model, train_loader=train_loader)
    budget_tracker = getattr(args, "budget_tracker", None)
    if budget_tracker is not None:
        budget_tracker.cum_macs += calib_costs["calibration_macs"]
        budget_tracker.cum_fwd_macs += calib_costs["calibration_macs"]
        budget_tracker.samples_seen += calib_costs["num_calib_samples"]

    # 2. Converti in SNN
    snn_model = convert_mlp_to_snn(ann_model, train_loader, device, mode=mode)
    snn_model.input_dim = ann_model.input_dim

    # Setup profiler neuromorfico SNN (traccia spike, firing rate, sparsity, SOPs, energia)
    snn_prof = SNNProfiler(model=snn_model, tag="mlp_snn", output_dir=RESULTS_DIR, num_steps=T)

    # 3. Valuta la SNN
    args.edge_profiler.stop()

    snn_acc = evaluate_mlp_snn(snn_model, test_loader, device, T=T, snn_prof=snn_prof)

    # Calcolo ed esportazione metriche neuromorfiche
    snn_prof.log_epoch("test", tag="test")
    snn_prof.print_summary()
    snn_prof.export_csv()

    profiler.log_eval(0.0, snn_acc)

    # 4. Salva il checkpoint del modello SNN convertito
    checkpoint_path = CHECKPOINT_DIR / checkpoint_out
    torch.save(snn_model.state_dict(), checkpoint_path)
    print(f"[MLP-SNN] Checkpoint salvato in: {checkpoint_path}")

    return snn_model
