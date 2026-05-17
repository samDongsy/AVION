"""AVION training entrypoint (few-shot and base-to-novel)."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    RSClassificationDataset,
    Sample,
    base_novel_split,
    deterministic_split,
    few_shot_sample,
    filter_by_class,
    list_image_folder,
    relabel_in_place,
)
from .losses import LossWeights, total_loss
from .model import PromptCLIP, StudentConfig
from .prototype import AggregationConfig
from .teacher import TeacherCLIP, TeacherConfig
from .utils import device_auto, load_yaml, set_seed


# --------------------------------------------------------------------- helpers
def build_loaders(
    train_samples: List[Sample], test_samples: List[Sample], student: PromptCLIP, batch_size: int
):
    transform = student.preprocess
    train_ds = RSClassificationDataset(train_samples, transform)
    test_ds = RSClassificationDataset(test_samples, transform)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    return train_loader, test_loader


def precompute_teacher_image_features(
    teacher: TeacherCLIP, samples: List[Sample], device: torch.device, batch_size: int = 32
) -> dict[str, torch.Tensor]:
    """Cache teacher visual features for the training set (used in L_img)."""
    ds = RSClassificationDataset(samples, teacher.preprocess)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    feats = {}
    for batch in tqdm(loader, desc="cache teacher img"):
        imgs = batch["image"].to(device)
        with torch.no_grad():
            f = teacher.encode_image(imgs).cpu()
        for j, p in enumerate(batch["path"]):
            feats[p] = f[j]
    return feats


@torch.no_grad()
def evaluate_accuracy(
    student: PromptCLIP,
    classnames: Sequence[str],
    loader: DataLoader,
    device: torch.device,
) -> float:
    student.eval()
    token_ids = student.tokenize_classes(classnames).to(device)
    text_emb = student.encode_text(token_ids)
    correct, total = 0, 0
    for batch in loader:
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device)
        v = student.encode_image(imgs)
        logits = v @ text_emb.t()
        pred = logits.argmax(-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    student.train()
    return 100.0 * correct / max(1, total)


# -------------------------------------------------------------------- training
def train_one_run(cfg: dict) -> dict:
    device = device_auto()
    set_seed(cfg.get("seed", 1))

    # ---- data ----
    samples, classnames = list_image_folder(cfg["data_root"])
    train_all, val_all, test_all = deterministic_split(samples, seed=cfg.get("seed", 1))

    if cfg["protocol"] == "few_shot":
        train_classes = classnames
        eval_classes = classnames
        train_samples = few_shot_sample(train_all, k=cfg["k_shot"], seed=cfg.get("seed", 1))
        eval_samples = test_all
        base_classes: Optional[List[str]] = None
    elif cfg["protocol"] == "base_to_novel":
        base, novel = base_novel_split(classnames, seed=cfg.get("seed", 1))
        train_classes = base
        train_samples = few_shot_sample(filter_by_class(train_all, base), k=16, seed=cfg.get("seed", 1))
        train_samples = relabel_in_place(train_samples, base)
        if cfg.get("eval_split", "base") == "base":
            eval_classes = base
            eval_samples = relabel_in_place(filter_by_class(test_all, base), base)
        else:
            eval_classes = novel
            eval_samples = relabel_in_place(filter_by_class(test_all, novel), novel)
        base_classes = base
    else:
        raise ValueError(f"unknown protocol: {cfg['protocol']}")

    # ---- student ----
    student_cfg = StudentConfig(**cfg["student"])
    student = PromptCLIP(student_cfg).to(device)
    train_loader, eval_loader = build_loaders(train_samples, eval_samples, student, cfg["batch_size"])

    # ---- teacher (only for L_img precaching; prototypes already in offline cache) ----
    teacher = TeacherCLIP(TeacherConfig(**cfg["teacher"])).to(device)
    teacher_img_feats = precompute_teacher_image_features(teacher, train_samples, device)

    # ---- offline cache (visual + text prototypes for training classes) ----
    cache = torch.load(cfg["offline_cache"], map_location="cpu")
    cache_classnames = cache["classnames"]
    cache_text = cache["text_prototypes"]
    # Re-order to match training classes.
    idx = [cache_classnames.index(c) for c in train_classes]
    t_T = cache_text[idx].to(device)        # (C_train, D)

    # ---- optimizer ----
    params = [p for p in student.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    weights = LossWeights(**cfg.get("loss_weights", {}))

    n_epochs = cfg["epochs"]
    total_steps = max(1, n_epochs * len(train_loader))
    step = 0

    token_ids = student.tokenize_classes(train_classes).to(device)
    teacher_text_logit_scale = teacher.model.logit_scale.exp().item()

    for epoch in range(n_epochs):
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            # Fresh student text features each step (prompts are learnable).
            t_S = student.encode_text(token_ids)               # (C, D)
            v_S = student.encode_image(imgs)                   # (B, D)
            student_logits = student.logit_scale.exp() * v_S @ t_S.t()

            # Teacher visual features from cache.
            v_T = torch.stack([teacher_img_feats[p] for p in batch["path"]]).to(device)
            teacher_logits = teacher_text_logit_scale * v_T @ t_T.t()

            loss, info = total_loss(
                student_logits=student_logits,
                labels=labels,
                v_S=v_S, v_T=v_T,
                t_S=t_S, t_T=t_T,
                teacher_logits=teacher_logits,
                weights=weights,
                step=step, total_steps=total_steps,
                base_class_mask=None,   # we already restrict classes via the index slice
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            step += 1

        # ---- eval at end of each epoch ----
        acc = evaluate_accuracy(student, eval_classes, eval_loader, device)
        print(f"[epoch {epoch}] loss={loss.item():.4f} eval_acc={acc:.2f}  ({info})")

    out_dir = Path(cfg.get("output_dir", "runs/avion"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "student_state": {k: v.cpu() for k, v in student.state_dict().items() if "prompt" in k or "logit_scale" in k},
        "classnames": list(train_classes),
        "config": cfg,
    }
    torch.save(ckpt, out_dir / "student.pt")
    return {"final_acc": acc}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    metrics = train_one_run(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
