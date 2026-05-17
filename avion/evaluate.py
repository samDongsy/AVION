"""Evaluation entrypoints: classification (few-shot, base-to-novel HM) and retrieval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    RSClassificationDataset,
    base_novel_split,
    deterministic_split,
    filter_by_class,
    list_image_folder,
    relabel_in_place,
)
from .model import PromptCLIP, StudentConfig
from .retrieval import (
    RetrievalCaptionDataset,
    RetrievalImageDataset,
    load_retrieval_annotations,
    mean_recall,
    recall_at_k,
)
from .train import evaluate_accuracy
from .utils import device_auto, load_yaml


def harmonic_mean(a: float, b: float) -> float:
    return (2 * a * b) / max(1e-9, (a + b))


def _load_student(ckpt_path: str, student_cfg: StudentConfig, device: torch.device) -> PromptCLIP:
    student = PromptCLIP(student_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    sd = state["student_state"] if "student_state" in state else state
    student.load_state_dict(sd, strict=False)
    student.eval()
    return student


# ----------------------------------------------------------------- classification
def eval_few_shot(cfg: dict) -> dict:
    device = device_auto()
    samples, classnames = list_image_folder(cfg["data_root"])
    _, _, test_all = deterministic_split(samples, seed=cfg.get("seed", 1))
    student = _load_student(cfg["ckpt"], StudentConfig(**cfg["student"]), device)
    test_ds = RSClassificationDataset(test_all, student.preprocess)
    loader = DataLoader(test_ds, batch_size=64, num_workers=2)
    acc = evaluate_accuracy(student, classnames, loader, device)
    return {"top1": acc}


def eval_base_to_novel(cfg: dict) -> dict:
    device = device_auto()
    samples, classnames = list_image_folder(cfg["data_root"])
    _, _, test_all = deterministic_split(samples, seed=cfg.get("seed", 1))
    base, novel = base_novel_split(classnames, seed=cfg.get("seed", 1))
    student = _load_student(cfg["ckpt"], StudentConfig(**cfg["student"]), device)

    def _acc(class_subset: List[str]) -> float:
        sub = relabel_in_place(filter_by_class(test_all, class_subset), class_subset)
        ds = RSClassificationDataset(sub, student.preprocess)
        loader = DataLoader(ds, batch_size=64, num_workers=2)
        return evaluate_accuracy(student, class_subset, loader, device)

    base_acc = _acc(base)
    novel_acc = _acc(novel)
    return {"base": base_acc, "novel": novel_acc, "HM": harmonic_mean(base_acc, novel_acc)}


# -------------------------------------------------------------------- retrieval
@torch.no_grad()
def eval_retrieval(cfg: dict) -> dict:
    device = device_auto()
    student = _load_student(cfg["ckpt"], StudentConfig(**cfg["student"]), device)

    items, _ = load_retrieval_annotations(cfg["data_root"])
    img_ds = RetrievalImageDataset(items, student.preprocess)
    cap_ds = RetrievalCaptionDataset(items)
    img_loader = DataLoader(img_ds, batch_size=64, num_workers=2)
    cap_loader = DataLoader(cap_ds, batch_size=64, num_workers=2)

    img_feats, img_ids = [], []
    for batch in tqdm(img_loader, desc="img"):
        v = student.encode_image(batch["image"].to(device))
        img_feats.append(v.cpu()); img_ids.append(batch["image_id"])
    img_feats = torch.cat(img_feats); img_ids = torch.cat(img_ids)

    txt_feats, txt_ids = [], []
    for batch in tqdm(cap_loader, desc="txt"):
        tok = student.tokenizer(list(batch["caption"])).to(device)
        t = student.encode_text(tok)
        txt_feats.append(t.cpu()); txt_ids.append(batch["image_id"])
    txt_feats = torch.cat(txt_feats); txt_ids = torch.cat(txt_ids)

    # T -> I: each caption's gt image id should appear in top-k images.
    sim_t2i = txt_feats @ img_feats.t()
    t2i = recall_at_k(sim_t2i, txt_ids, img_ids)

    # I -> T: each image should retrieve any of its captions in top-k.
    sim_i2t = img_feats @ txt_feats.t()
    i2t = recall_at_k(sim_i2t, img_ids, txt_ids)

    return {
        "T->I": t2i,
        "I->T": i2t,
        "mR": mean_recall(t2i, i2t),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--task", choices=["few_shot", "base_to_novel", "retrieval"], required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    if args.task == "few_shot":
        out = eval_few_shot(cfg)
    elif args.task == "base_to_novel":
        out = eval_base_to_novel(cfg)
    else:
        out = eval_retrieval(cfg)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
