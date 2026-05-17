"""Frozen teacher CLIP wrapper (e.g., GeoRSCLIP ViT-H/14)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TeacherConfig:
    backbone: str = "ViT-H-14"
    pretrained: str = "laion2b_s32b_b79k"
    pretrained_path: Optional[str] = None
    device: str = "cuda"


class TeacherCLIP(nn.Module):
    def __init__(self, cfg: TeacherConfig):
        super().__init__()
        self.cfg = cfg
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.backbone, pretrained=cfg.pretrained
        )
        if cfg.pretrained_path:
            sd = torch.load(cfg.pretrained_path, map_location="cpu")
            sd = sd.get("state_dict", sd)
            model.load_state_dict(sd, strict=False)
        self.tokenizer = open_clip.get_tokenizer(cfg.backbone)
        self.preprocess = preprocess
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(images)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        tok = self.tokenizer(list(texts)).to(next(self.model.parameters()).device)
        feats = self.model.encode_text(tok)
        return F.normalize(feats, dim=-1)
