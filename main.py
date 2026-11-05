
from argument import get_args
from datasets.dataset import get_dataloaders
from device import get_device
from paradigm import get_paradigm
from paths import RESULTS_DIR
from profiler import EdgeProfiler, TrainEvalProfiler, BudgetTracker
import logging
from utils import get_info_dataset

logging.getLogger("torch._library.triton").setLevel(logging.ERROR)

def main(args):
    paradigm = args.paradigm
    paradigm_args = getattr(args, paradigm, None)
    if paradigm_args is None:
        raise ValueError(f"Sezione '{paradigm}' mancante nel config.")

    run_fn = get_paradigm(paradigm)

    train_loader, test_loader = get_dataloaders(args)

    aug_flag,norm_flag,n_train,n_test,n_total = get_info_dataset(args, train_loader, test_loader)
    print(
        f"Dataset: {args.dataset.name} | Normalization: {norm_flag} | Augmentation: {aug_flag} | "
        f"Campioni totali: {n_total} (Train: {n_train}, Test: {n_test}) | "
        f"Campioni per batch -> Train: {train_loader.batch_size}, Test: {test_loader.batch_size}"
    )

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

    # Setup Budget Tracker (controllo budget computazionale fisso)
    b_cfg = getattr(args, "budget", None)
    budget_tracker = BudgetTracker(
        target_macs=getattr(b_cfg, "max_macs", None),
        enabled=getattr(b_cfg, "enabled", False),
        paradigm=paradigm,
        tag=tag,
        output_dir=RESULTS_DIR,
    ) if b_cfg else None

    if budget_tracker:
        paradigm_args.budget_tracker = budget_tracker
        if train_prof is None and budget_tracker.enabled:
            train_prof = TrainEvalProfiler(tag=tag, output_dir=RESULTS_DIR)
        if train_prof:
            train_prof.budget_tracker = budget_tracker

    if edge_prof:
        edge_prof.start()

    try:
        # ESEGUE IL PARADIGMA SELEZIONATO (TRAIN + EVAL)
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

        if budget_tracker and (budget_tracker.enabled or budget_tracker.cum_macs > 0):
            budget_tracker.print_summary()
            if getattr(b_cfg, "export_csv", True):
                budget_tracker.export_csv()


if __name__ == "__main__":
    main(get_args())
