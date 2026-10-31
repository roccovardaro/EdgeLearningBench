
from argument import get_args
from datasets.dataset import get_dataloaders
from device import get_device
from paths import RESULTS_DIR
from profiler import EdgeProfiler, TrainEvalProfiler
from models.forward_forward.ff_mlp import ff_mlp_run
from models.forward_forward.ff_resnet import ff_resnet_run
from models.baseline.resnet import resnet_run
from models.baseline.cnn import cnn_run
from models.kan.kan import kan_run
from models.kan.direct_kan import direct_kan_run
from models.kan.cnn_kan import cnn_kan_run
from models.baseline.mlp import mlp_run
from models.spiking_neural_network.snn_mlp import snn_mlp_run
from models.spiking_neural_network.snn_cnn import snn_cnn_run
from models.spiking_neural_network.snn_resnet import snn_resnet_run
from models.spiking_neural_network.snn_stdp import snn_stdp_run
from models.spiking_neural_network.ann_snn_conversion.mlp_snn import mlp_snn_run
from models.spiking_neural_network.ann_snn_conversion.cnn_snn import cnn_snn_run
from models.ltc.ltc import ltc_run
from models.ltc.cnn_ltc import cnn_ltc_run
from models.hebbian.chl import chl_run

_PARADIGMS = {
    "ff_mlp": ff_mlp_run,
    "ff_resnet": ff_resnet_run,
    "resnet": resnet_run,
    "cnn": cnn_run,
    "kan": kan_run,
    "direct_kan": direct_kan_run,
    "cnn_kan": cnn_kan_run,
    "mlp": mlp_run,
    "snn_mlp": snn_mlp_run,
    "snn_cnn": snn_cnn_run,
    "snn_resnet": snn_resnet_run,
    "snn_stdp": snn_stdp_run,
    "mlp_snn": mlp_snn_run,
    "cnn_snn": cnn_snn_run,
    "ltc": ltc_run,
    "cnn_ltc": cnn_ltc_run,
    "chl": chl_run,
}


def main(args):
    paradigm = args.paradigm
    paradigm_args = getattr(args, paradigm, None)
    if paradigm_args is None:
        raise ValueError(f"Sezione '{paradigm}' mancante nel config.")

    run_fn = _PARADIGMS.get(paradigm)
    if run_fn is None:
        raise ValueError(f"Paradigma '{paradigm}' non riconosciuto. Opzioni: {', '.join(_PARADIGMS)}")

    train_loader, test_loader = get_dataloaders(args)
    aug_flag = getattr(args.dataset, "augmentation", False)
    norm_flag = getattr(args.dataset, "normalize", True)
    print(f"Dataset: {args.dataset.name} | Normalization: {norm_flag} | Augmentation: {aug_flag} | Batch size -> Train: {train_loader.batch_size}, Test: {test_loader.batch_size}")
    tag = f"{paradigm}_{args.dataset.name}"

    device = get_device(getattr(args, "device", "auto"))
    print(f"Device: {device}")

    # Setup profiler (hardware e metriche)
    p_cfg = getattr(args, "profiler", None)
    edge_prof = EdgeProfiler(
        interval_sec=getattr(p_cfg, "interval_sec", 1.0),
        track_energy=getattr(p_cfg, "track_energy", True),
        tag=tag,
        output_dir=RESULTS_DIR,
    ) if getattr(p_cfg, "enabled", False) else None

    te_cfg = getattr(args, "train_eval_profiler", None)
    train_prof = TrainEvalProfiler(tag=tag, output_dir=RESULTS_DIR) if getattr(te_cfg, "enabled", False) else None

    if edge_prof:
        edge_prof.start()

    try:
        #ESEGUE IL PARADIGMA SELEZIONATO (TRAIN + EVAL)
        run_fn(paradigm_args, train_loader, test_loader, profiler=train_prof, device=device)
    finally:
        if edge_prof:
            edge_prof.stop()
            edge_prof.print_summary()
            if getattr(p_cfg, "export_csv", True):
                edge_prof.export_csv()
            if getattr(p_cfg, "export_json", True):
                edge_prof.export_json()

        if train_prof and getattr(te_cfg, "export_csv", True):
            train_prof.export_csv()


if __name__ == "__main__":
    main(get_args())
