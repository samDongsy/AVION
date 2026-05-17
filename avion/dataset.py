"""RS classification datasets + few-shot and base-to-novel sampling.

All datasets are expected to follow the standard ImageFolder layout::

    root/
      class_a/
        img1.jpg
        img2.jpg
        ...
      class_b/
        ...

The split files (train/val/test) follow the CoOp protocol if present
(``split.json``); otherwise we fall back to a deterministic per-class
70/10/20 split with a fixed seed.

Supported datasets (with default class counts taken from AVION Sec. 4.1):
  AID(30), RESISC45(45), EuroSAT(10), WHU-RS19(19), PatternNet(38),
  UCMerced(21)
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Sample:
    path: str
    label: int
    classname: str


def list_image_folder(root: str | Path) -> Tuple[List[Sample], List[str]]:
    root = Path(root)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    cls2idx = {c: i for i, c in enumerate(classes)}
    samples: List[Sample] = []
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    for c in classes:
        for p in sorted((root / c).iterdir()):
            if p.suffix.lower() in exts:
                samples.append(Sample(str(p), cls2idx[c], c))
    return samples, classes


def deterministic_split(
    samples: List[Sample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    seed: int = 1,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Per-class shuffled split with a fixed seed."""
    by_class: Dict[int, List[Sample]] = defaultdict(list)
    for s in samples:
        by_class[s.label].append(s)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for label, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_tr = int(n * train_ratio)
        n_va = int(n * val_ratio)
        train.extend(items[:n_tr])
        val.extend(items[n_tr : n_tr + n_va])
        test.extend(items[n_tr + n_va :])
    return train, val, test


def few_shot_sample(samples: List[Sample], k: int, seed: int = 1) -> List[Sample]:
    """Sample k images per class deterministically."""
    rng = random.Random(seed)
    by_class: Dict[int, List[Sample]] = defaultdict(list)
    for s in samples:
        by_class[s.label].append(s)
    out: List[Sample] = []
    for label, items in by_class.items():
        items = sorted(items, key=lambda s: s.path)
        rng.shuffle(items)
        out.extend(items[:k])
    return out


def base_novel_split(
    classnames: Sequence[str], seed: int = 1, base_ratio: float = 0.5
) -> Tuple[List[str], List[str]]:
    """Half/half base-novel split with a fixed seed."""
    rng = random.Random(seed)
    idx = list(range(len(classnames)))
    rng.shuffle(idx)
    n_base = max(1, int(len(idx) * base_ratio))
    base = sorted(classnames[i] for i in idx[:n_base])
    novel = sorted(classnames[i] for i in idx[n_base:])
    return base, novel


def filter_by_class(samples: List[Sample], classnames: Sequence[str]) -> List[Sample]:
    keep = set(classnames)
    return [s for s in samples if s.classname in keep]


def relabel_in_place(samples: List[Sample], classnames: Sequence[str]) -> List[Sample]:
    """Re-label samples to indices in ``classnames``."""
    mapping = {c: i for i, c in enumerate(classnames)}
    out = []
    for s in samples:
        if s.classname in mapping:
            out.append(Sample(s.path, mapping[s.classname], s.classname))
    return out


# ----------------------------------------------------------------- Dataset
class RSClassificationDataset(Dataset):
    def __init__(self, samples: List[Sample], transform: Callable):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        img = self.transform(img)
        return {"image": img, "label": s.label, "classname": s.classname, "path": s.path}
