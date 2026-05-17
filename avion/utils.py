"""Small utilities: config loading, seeding, logging."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = -1.0


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_module(m: torch.nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad_(False)
