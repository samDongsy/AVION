"""Offline preparation: cache teacher visual prototypes and teacher text prototypes.

Run once per (dataset, teacher) pair before training. Saves a ``.pt`` file
holding the per-class teacher visual and text prototypes, ready to be loaded
by the training loop.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import RSClassificationDataset, Sample
from .llm_prompting import load_class_descriptions
from .prototype import AggregationConfig, build_class_prototypes
from .teacher import TeacherCLIP, TeacherConfig


@torch.no_grad()
def compute_visual_prototypes(
    teacher: TeacherCLIP,
    samples: List[Sample],
    classnames: Sequence[str],
    transform,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 2,
) -> torch.Tensor:
    """Return per-class mean of ℓ2-normalised teacher visual embeddings (C, D)."""
    ds = RSClassificationDataset(samples, transform)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    C = len(classnames)
    sums = None
    counts = torch.zeros(C, dtype=torch.long)
    for batch in tqdm(loader, desc="teacher visual"):
        imgs = batch["image"].to(device)
        feats = teacher.encode_image(imgs).cpu()      # already ℓ2-normalized
        labels = batch["label"]
        if sums is None:
            sums = torch.zeros(C, feats.shape[-1])
        sums.index_add_(0, labels, feats)
        counts.index_add_(0, labels, torch.ones_like(labels))
    counts = counts.clamp_min(1).unsqueeze(-1).float()
    proto = sums / counts
    proto = proto / proto.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return proto


@torch.no_grad()
def compute_text_prototypes(
    teacher: TeacherCLIP,
    classnames: Sequence[str],
    class_descriptions: Dict[str, List[str]],
    device: torch.device,
    visual_prototypes: torch.Tensor,
    agg_cfg: AggregationConfig,
) -> torch.Tensor:
    """Run selective prototype aggregation, returning per-class text prototypes."""
    visual_prototypes = visual_prototypes.to(device)
    class_caption_embeds: List[torch.Tensor] = []
    class_captions: List[List[str]] = []
    for k, name in enumerate(classnames):
        caps = class_descriptions.get(name, [])
        if not caps:                                 # fall back to a manual template
            caps = [f"a satellite photo of a {name.replace('_', ' ')}."]
        emb = teacher.encode_text(caps).to(device)
        class_captions.append(caps)
        class_caption_embeds.append(emb)
    protos = build_class_prototypes(
        visual_prototypes, class_captions, class_caption_embeds, agg_cfg
    )
    return protos.cpu()


def prepare_offline_cache(
    *,
    teacher_cfg: TeacherConfig,
    samples: List[Sample],
    classnames: Sequence[str],
    descriptions_json: str,
    out_path: str,
    agg_cfg: AggregationConfig = AggregationConfig(),
    batch_size: int = 32,
    num_workers: int = 2,
) -> None:
    device = torch.device(teacher_cfg.device)
    teacher = TeacherCLIP(teacher_cfg).to(device).eval()
    transform = teacher.preprocess
    descriptions = load_class_descriptions(descriptions_json)
    vis = compute_visual_prototypes(
        teacher, samples, classnames, transform, device, batch_size, num_workers
    )
    txt = compute_text_prototypes(teacher, classnames, descriptions, device, vis, agg_cfg)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classnames": list(classnames),
            "visual_prototypes": vis,    # (C, D)
            "text_prototypes": txt,      # (C, D)
            "teacher_backbone": teacher_cfg.backbone,
        },
        out_path,
    )
    print(f"saved offline cache to {out_path} (C={len(classnames)})")


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--descriptions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--teacher-backbone", default="ViT-H-14")
    p.add_argument("--teacher-pretrained", default="laion2b_s32b_b79k")
    p.add_argument("--teacher-ckpt", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from .dataset import list_image_folder
    samples, classnames = list_image_folder(args.data_root)
    teacher_cfg = TeacherConfig(
        backbone=args.teacher_backbone,
        pretrained=args.teacher_pretrained,
        pretrained_path=args.teacher_ckpt,
        device=args.device,
    )
    prepare_offline_cache(
        teacher_cfg=teacher_cfg,
        samples=samples,
        classnames=classnames,
        descriptions_json=args.descriptions,
        out_path=args.out,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    _cli()
