"""Selective prototype aggregation: RS-Flag rules, MAD pruning, weighted aggregation.

Implements Sec. 3.2 of AVION (and Appendix D pseudo-code).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------- RS-Flag
RS_POSITIVE = {
    "overhead", "aerial view", "satellite imagery", "nadir",
    "orthorectified", "multispectral", "sar",
}
RS_NEGATIVE = {
    "street", "indoor", "selfie", "portrait", "close-up", "ground level",
}


def _contains_phrase(text: str, phrase: str) -> bool:
    """Case-insensitive whole-word/phrase match."""
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pattern, text, re.IGNORECASE) is not None


def rs_flag(caption: str, min_words: int = 6, max_words: int = 20) -> int:
    """Return 1 iff the caption satisfies the RS-Flag rules (Appendix D)."""
    n_words = len(caption.split())
    if not (min_words <= n_words <= max_words):
        return 0
    has_pos = any(_contains_phrase(caption, p) for p in RS_POSITIVE)
    has_neg = any(_contains_phrase(caption, p) for p in RS_NEGATIVE)
    return int(has_pos and not has_neg)


# --------------------------------------------------------------- aggregation
@dataclass
class AggregationConfig:
    beta: float = 10.0          # similarity weight in eq. 3
    gamma: float = 2.0          # RS-flag calibration weight
    zeta_s: float = 3.0         # MAD-pruning threshold
    eps: float = 1e-8           # numerical stability
    use_rs_flag: bool = True    # set False for general-domain (e.g., ImageNet)


def _robust_z_scores(scores: torch.Tensor, eps: float) -> torch.Tensor:
    med = scores.median()
    mad = (scores - med).abs().median()
    return (scores - med).abs() / (mad + eps)


def selective_prototype_aggregation(
    visual_prototype: torch.Tensor,         # (D,) ℓ2-normalized teacher visual prototype
    caption_embeds: torch.Tensor,           # (J, D) ℓ2-normalized teacher text embeddings
    captions: Sequence[str],                # raw caption strings (for RS-flag)
    cfg: AggregationConfig = AggregationConfig(),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Aggregate caption embeddings into a single class prototype.

    Returns (prototype (D,), info dict with kept indices and weights).
    """
    if caption_embeds.numel() == 0:
        raise ValueError("No caption embeddings provided.")
    J = caption_embeds.shape[0]
    # Step 1: similarity scores s_{k,j} = v_hat^T t_{k,j}.
    scores = caption_embeds @ visual_prototype  # (J,)

    # Step 2-3: MAD-based robust pruning. Keep candidates with z <= zeta_s.
    z = _robust_z_scores(scores, cfg.eps)
    keep_mask = z <= cfg.zeta_s
    if keep_mask.sum() == 0:        # fall back to all candidates
        keep_mask = torch.ones_like(keep_mask)

    kept_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
    kept_scores = scores[kept_idx]
    kept_embeds = caption_embeds[kept_idx]
    if cfg.use_rs_flag:
        flags = torch.tensor(
            [rs_flag(captions[int(j)]) for j in kept_idx.tolist()],
            dtype=scores.dtype, device=scores.device,
        )
    else:
        flags = torch.zeros_like(kept_scores)

    # Step 4: softmax-normalised weights with beta/gamma calibration (eq. 3).
    logits = cfg.beta * kept_scores + cfg.gamma * flags
    weights = torch.softmax(logits, dim=0)

    # Step 5: weighted aggregation and ℓ2-normalise.
    prototype = (weights.unsqueeze(-1) * kept_embeds).sum(dim=0)
    prototype = F.normalize(prototype, dim=-1)
    return prototype, {
        "kept_idx": kept_idx.cpu(),
        "weights": weights.cpu(),
        "scores": scores.cpu(),
        "z": z.cpu(),
        "flags": flags.cpu(),
    }


# --------------------------------------------------------------- batch helper
def build_class_prototypes(
    visual_prototypes: torch.Tensor,        # (C, D)
    class_captions: List[List[str]],        # length C; each is list[str]
    class_caption_embeds: List[torch.Tensor],  # length C; each (J_k, D)
    cfg: AggregationConfig = AggregationConfig(),
) -> torch.Tensor:
    """Return per-class teacher prototypes (C, D), ℓ2-normalized."""
    C = visual_prototypes.shape[0]
    assert len(class_captions) == C == len(class_caption_embeds)
    out = []
    for k in range(C):
        proto, _ = selective_prototype_aggregation(
            visual_prototypes[k], class_caption_embeds[k], class_captions[k], cfg
        )
        out.append(proto)
    return torch.stack(out, dim=0)
