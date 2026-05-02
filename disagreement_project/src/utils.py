"""Utility helpers: seeding, config loading, checkpointing, device selection."""
from __future__ import annotations

import os
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed everything for reproducibility.

    When deterministic=True:
      - cuDNN benchmark is disabled (slower but reproducible).
      - CUBLAS_WORKSPACE_CONFIG is set and torch.use_deterministic_algorithms
        is enabled so that atomic CUDA ops (e.g. scatter_add in embedding
        layers) are also deterministic on PyTorch ≥ 1.11.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Required for cuBLAS determinism on CUDA ≥ 10.2
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            # torch < 1.8 does not have this function
            pass
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ("data_root", "checkpoints", "figures", "tables", "logs"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location=None) -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class EarlyStopper:
    """Minimum-validation-loss early stopper."""
    patience: int = 8
    min_delta: float = 0.0
    best: float = float("inf")
    counter: int = 0
    should_stop: bool = False

    def step(self, current: float) -> bool:
        """Return True if this is a new best."""
        if current < self.best - self.min_delta:
            self.best = current
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
