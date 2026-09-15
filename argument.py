import argparse
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
    cli_args = parser.parse_args()
    with open(cli_args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    args = to_namespace(config)
    return args