"""Cross-modal retrieval: RSITMD / RSICD style image-caption datasets + metrics.

Expected layout::

    root/
      images/  # all images
      annotations.json   # list of {"image": "img_001.jpg", "captions": [...]}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class RetrievalItem:
    image_path: str
    captions: List[str]
    image_id: int


def load_retrieval_annotations(root: str | Path) -> Tuple[List[RetrievalItem], List[str]]:
    root = Path(root)
    with open(root / "annotations.json", "r", encoding="utf-8") as f:
        items_raw = json.load(f)
    items, captions = [], []
    for i, x in enumerate(items_raw):
        img_path = str(root / "images" / x["image"])
        caps = list(x["captions"])
        items.append(RetrievalItem(img_path, caps, i))
        captions.extend(caps)
    return items, captions


class RetrievalImageDataset(Dataset):
    def __init__(self, items: List[RetrievalItem], transform: Callable):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        x = self.items[i]
        img = Image.open(x.image_path).convert("RGB")
        return {"image": self.transform(img), "image_id": x.image_id}


class RetrievalCaptionDataset(Dataset):
    """Yields one caption per index along with its associated image id."""

    def __init__(self, items: List[RetrievalItem]):
        self.captions: List[str] = []
        self.image_ids: List[int] = []
        for x in items:
            for c in x.captions:
                self.captions.append(c)
                self.image_ids.append(x.image_id)

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, i: int):
        return {"caption": self.captions[i], "image_id": self.image_ids[i]}


@torch.no_grad()
def recall_at_k(
    sim: torch.Tensor,                    # (Nq, Ng)
    gt_ids: torch.Tensor,                 # (Nq,) integer ids
    gallery_ids: torch.Tensor,            # (Ng,) integer ids
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute R@K for a similarity matrix. Multiple gallery items may map to the same id."""
    out = {}
    topk = sim.topk(max(ks), dim=-1).indices         # (Nq, Kmax)
    matched = gallery_ids[topk] == gt_ids.unsqueeze(-1)
    for k in ks:
        hit = matched[:, :k].any(dim=-1).float()
        out[f"R@{k}"] = 100.0 * hit.mean().item()
    return out


def mean_recall(t2i: Dict[str, float], i2t: Dict[str, float]) -> float:
    return (sum(t2i.values()) + sum(i2t.values())) / (len(t2i) + len(i2t))
