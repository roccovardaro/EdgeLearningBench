
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
        print(f"Paradigma '{paradigm}' mancante nel config.")
        return

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

    # Setup profiler (sempre attivi nel progetto)
    p_cfg = getattr(args, "profiler", None)
    edge_prof = EdgeProfiler(
        interval_sec=getattr(p_cfg, "interval_sec", 1.0) if p_cfg else 1.0,
        track_energy=getattr(p_cfg, "track_energy", True) if p_cfg else True,
        tag=tag,
        output_dir=RESULTS_DIR,
    )
    train_prof = TrainEvalProfiler(tag=tag, output_dir=RESULTS_DIR)

    # Setup Budget Tracker (controllo budget computazionale fisso)
    b_cfg = getattr(args, "budget", None)
    budget_enabled = getattr(b_cfg, "enabled", False) and (getattr(b_cfg, "max_macs", 0) or 0) > 0
    budget_tracker = BudgetTracker(
        target_macs=getattr(b_cfg, "max_macs", None),
        enabled=True,
        paradigm=paradigm,
        tag=tag,
        output_dir=RESULTS_DIR,
    ) if (b_cfg and budget_enabled) else None

    if budget_tracker:
        paradigm_args.budget_tracker = budget_tracker
        train_prof.budget_tracker = budget_tracker
    else:
        paradigm_args.budget_tracker = None

    paradigm_args.edge_profiler = edge_prof

    edge_prof.start()

    try:
        # ESEGUE IL PARADIGMA SELEZIONATO (TRAIN + EVAL)
        run_fn(paradigm_args, train_loader, test_loader, profiler=train_prof, device=device)
    finally:
        edge_prof.stop()
        edge_prof.print_summary()
        if getattr(p_cfg, "export_csv", True) if p_cfg else True:
            edge_prof.export_csv()
        if getattr(p_cfg, "export_json", True) if p_cfg else True:
            edge_prof.export_json()

        train_prof.export_csv()

        if budget_tracker and budget_tracker.enabled:
            budget_tracker.print_summary()
            if getattr(b_cfg, "export_csv", True):
                budget_tracker.export_csv()


if __name__ == "__main__":
    main(get_args())
