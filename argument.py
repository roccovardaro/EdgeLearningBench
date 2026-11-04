import argparse
from pathlib import Path
from types import SimpleNamespace
import yaml

DATASET_MAP = {
    "mnist": "configs/mnist.yaml",
    "cifar10": "configs/cifar10.yaml",
    "fashion_mnist": "configs/fashion_mnist.yaml",
    "adult": "configs/adult.yaml",
    "breast_cancer": "configs/breast_cancer.yaml",
    "wine_quality": "configs/wine_quality.yaml",
}


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: to_namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="mnist",
        help=f"Nome del dataset ({', '.join(DATASET_MAP.keys())}) o percorso del file .yaml (default: mnist)",
    )
    parser.add_argument("--profiler_config", default="configs/profiler.yaml")
    parser.add_argument("--device", default=None, help="Dispositivo di calcolo (cpu, mps, cuda, auto)")
    parser.add_argument("--paradigm", default=None, help="Paradigma di calcolo/modello")
    parser.add_argument(
        "--max_macs",
        type=float,
        default=None,
        help="Budget massimo di MAC equivalenti (es. 1e14 o 5e12). Se specificato, attiva automaticamente il budget.",
    )
    parser.add_argument(
        "--budget_enabled",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=None,
        help="Attiva o disattiva il controllo del budget computazionale fisso (true/false).",
    )
    cli_args = parser.parse_args()

    config_path = DATASET_MAP.get(cli_args.dataset, cli_args.dataset)
    config_file = Path(config_path)

    if not config_file.is_file():
        candidate = Path("configs") / f"{cli_args.dataset}.yaml"
        if candidate.is_file():
            config_file = candidate
        else:
            raise FileNotFoundError(
                f"Dataset '{cli_args.dataset}' non trovato. "
                f"Scegli tra {list(DATASET_MAP.keys())} oppure specifica un percorso valido."
            )

    with open(config_file, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    profiler_path = Path(cli_args.profiler_config)
    if profiler_path.is_file():
        with open(profiler_path, encoding="utf-8") as file:
            p_config = yaml.safe_load(file) or {}
            if isinstance(p_config, dict):
                for k, v in p_config.items():
                    config[k] = v

    if cli_args.device:
        config["device"] = cli_args.device
    if cli_args.paradigm:
        config["paradigm"] = cli_args.paradigm

    # Gestione e override del Budget computazionale
    if "budget" not in config or not isinstance(config["budget"], dict):
        config["budget"] = {
            "enabled": False,
            "max_macs": None,
            "include_optimizer": True,
            "export_csv": True,
        }

    # Assicura che max_macs sia float se specificato in YAML
    if config["budget"].get("max_macs") is not None:
        try:
            config["budget"]["max_macs"] = float(config["budget"]["max_macs"])
        except (ValueError, TypeError):
            pass

    # Override da CLI
    if cli_args.max_macs is not None:
        config["budget"]["max_macs"] = float(cli_args.max_macs)
        config["budget"]["enabled"] = True
    if cli_args.budget_enabled is not None:
        config["budget"]["enabled"] = cli_args.budget_enabled

    args = to_namespace(config)
    return args
