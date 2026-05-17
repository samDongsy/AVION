"""AVION tri-aspect alignment losses + task loss + λ scheduling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    lambda_img: float = 0.5
    lambda_text: float = 0.5
    lambda_logit: float = 1.0
    temperature: float = 2.0       # τ in eq. 7
    warmup_frac: float = 0.30      # 30% linear warm-up for λ_logit


def lambda_logit_schedule(weights: LossWeights, step: int, total_steps: int) -> float:
    """Linear warm-up of the logit-distillation weight over warmup_frac of training."""
    warm_steps = max(1, int(total_steps * weights.warmup_frac))
    if step >= warm_steps:
        return weights.lambda_logit
    return weights.lambda_logit * (step / warm_steps)


def task_loss(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy over class logits."""
    return F.cross_entropy(student_logits, labels)


def image_alignment_loss(v_S: torch.Tensor, v_T: torch.Tensor) -> torch.Tensor:
    """1 - cos(v_S, v_T) averaged over the batch (eq. 5)."""
    cos = (v_S * v_T).sum(dim=-1)
    return (1.0 - cos).mean()


def text_alignment_loss(t_S: torch.Tensor, t_T: torch.Tensor) -> torch.Tensor:
    """1 - cos(t_S, t_T*) averaged over classes (eq. 6)."""
    cos = (t_S * t_T).sum(dim=-1)
    return (1.0 - cos).mean()


def logit_distillation_loss(
    student_logits: torch.Tensor,           # (B, C)
    teacher_logits: torch.Tensor,           # (B, C)
    temperature: float = 2.0,
    base_class_mask: Optional[torch.Tensor] = None,  # (C,) bool or None
) -> torch.Tensor:
    """Temperature-scaled KL divergence between teacher and student (eq. 7).

    If ``base_class_mask`` is given (base-to-novel training), the logits are
    masked to base classes and renormalised (Appendix E).
    """
    if base_class_mask is not None:
        neg = torch.finfo(student_logits.dtype).min
        mask = base_class_mask.to(student_logits.device).unsqueeze(0)
        student_logits = student_logits.masked_fill(~mask, neg)
        teacher_logits = teacher_logits.masked_fill(~mask, neg)

    log_p_S = F.log_softmax(student_logits / temperature, dim=-1)
    p_T = F.softmax(teacher_logits / temperature, dim=-1)
    # KL(p_T || p_S) per sample, then mean.
    kl = F.kl_div(log_p_S, p_T, reduction="batchmean")
    return (temperature ** 2) * kl


def total_loss(
    *,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    v_S: torch.Tensor,
    v_T: torch.Tensor,
    t_S: torch.Tensor,
    t_T: torch.Tensor,
    teacher_logits: torch.Tensor,
    weights: LossWeights,
    step: int,
    total_steps: int,
    base_class_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compose the AVION objective (eq. 8) with logit-weight warm-up."""
    L_task = task_loss(student_logits, labels)
    L_img = image_alignment_loss(v_S, v_T)
    L_text = text_alignment_loss(t_S, t_T)
    L_logit = logit_distillation_loss(
        student_logits, teacher_logits, weights.temperature, base_class_mask
    )
    lam_logit = lambda_logit_schedule(weights, step, total_steps)
    loss = (
        L_task
        + weights.lambda_img * L_img
        + weights.lambda_text * L_text
        + lam_logit * L_logit
    )
    return loss, {
        "L_task": L_task.item(),
        "L_img": L_img.item(),
        "L_text": L_text.item(),
        "L_logit": L_logit.item(),
        "lambda_logit": lam_logit,
    }
