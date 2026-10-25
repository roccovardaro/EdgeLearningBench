import argparse
from pathlib import Path
from types import SimpleNamespace
import yaml


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
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--profiler_config", default="configs/profiler.yaml")
    parser.add_argument("--device", default=None, help="Dispositivo di calcolo (cpu, mps, cuda, auto)")
    parser.add_argument("--paradigm", default=None, help="Paradigma di calcolo/modello")
    cli_args = parser.parse_args()

    with open(cli_args.config, encoding="utf-8") as file:
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

    args = to_namespace(config)
    return args