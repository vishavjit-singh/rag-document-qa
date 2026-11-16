"""Settings from ragqa.yaml with a few environment overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RAGQA_HOME", Path(__file__).resolve().parents[2]))
ENV_OVERRIDES = {"RAGQA_EMBEDDER": ("embedding", "backend"), "RAGQA_INDEX": ("index", "backend"),
                 "RAGQA_ANSWERER": ("answer", "backend")}


def load(path: str | Path | None = None, root: Path | None = None) -> dict:
    root = Path(root or ROOT)
    cfg = yaml.safe_load(Path(path or root / "ragqa.yaml").read_text())
    for env, (section, key) in ENV_OVERRIDES.items():
        if os.environ.get(env):
            cfg[section][key] = os.environ[env]
    cfg["paths"] = {k: (root / v).resolve() for k, v in cfg["paths"].items()}
    return cfg


def with_overrides(cfg: dict, **sections) -> dict:
    """Copy of cfg with e.g. chunking={"size": 80} merged in."""
    out = copy.deepcopy(cfg)
    for section, values in sections.items():
        out[section].update({k: v for k, v in values.items() if v is not None})
    return out
