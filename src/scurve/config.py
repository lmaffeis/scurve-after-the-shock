"""Load YAML config; paths resolved relative to repo root."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = REPO_ROOT / "configs" / "base.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = REPO_ROOT / rel
    return cfg
