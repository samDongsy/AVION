"""AVION student CLIP with deep prompts on both vision and text branches.

Implementation notes
--------------------
We build the student on top of an open_clip backbone. The backbone weights are
frozen. We inject `Lv * Pv` learnable visual prompts and `Lt * Pt` learnable
text prompts (one set per transformer layer; VPT-deep / CoOp-style deep
prompts). The prompt tokens are inserted after the [CLS] / [SOS] position so
that pooling indices (cls, eos) remain unchanged. Prompt outputs from layer
`ell - 1` are *replaced* by fresh learnable prompts at the input of layer
`ell`, which is the standard "deep" prompt scheme used by VPT-Deep and MaPLe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StudentConfig:
    backbone: str = "ViT-B-32"
    pretrained: str = "openai"            # or path to GeoRSCLIP/RemoteCLIP ckpt
    n_vision_prompts: int = 8             # P_v
    n_text_prompts: int = 4               # P_t
    vision_prompt_depth: int = 12         # L_v (<= backbone vision depth)
    text_prompt_depth: int = 12           # L_t (<= backbone text depth)
    prompt_init_std: float = 0.02
    pretrained_path: Optional[str] = None  # if set, load state_dict from path


def _trunc_normal_(t: torch.Tensor, std: float) -> torch.Tensor:
    nn.init.trunc_normal_(t, std=std, a=-2 * std, b=2 * std)
    return t


class PromptCLIP(nn.Module):
    """CLIP backbone wrapper with VPT-deep prompts in vision + text branches."""

    def __init__(self, cfg: StudentConfig):
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
        self.model = model

        for p in self.model.parameters():
            p.requires_grad_(False)

        # Dimensions.
        # open_clip visual transformer width: visual.transformer.width
        self.vis_width = model.visual.transformer.width
        self.vis_depth = len(model.visual.transformer.resblocks)
        # Text transformer width
        self.txt_width = model.transformer.width
        self.txt_depth = len(model.transformer.resblocks)
        # ctx_length for tokenizer
        self.context_length = model.context_length if hasattr(model, "context_length") else 77

        Lv = min(cfg.vision_prompt_depth, self.vis_depth)
        Lt = min(cfg.text_prompt_depth, self.txt_depth)
        self.Lv, self.Lt = Lv, Lt
        self.Pv, self.Pt = cfg.n_vision_prompts, cfg.n_text_prompts

        self.vision_prompts = nn.Parameter(torch.empty(Lv, self.Pv, self.vis_width))
        self.text_prompts = nn.Parameter(torch.empty(Lt, self.Pt, self.txt_width))
        _trunc_normal_(self.vision_prompts, cfg.prompt_init_std)
        _trunc_normal_(self.text_prompts, cfg.prompt_init_std)

        # learnable logit scale (initialized to backbone's value)
        self.logit_scale = nn.Parameter(self.model.logit_scale.detach().clone())

    # ------------------------------------------------------------------ vision
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) -> (B, D) ℓ2-normalized embeddings."""
        v = self.model.visual
        # patch embed
        x = v.conv1(images)                                     # (B, D, H', W')
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, N, D)
        cls = v.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)                          # (B, 1+N, D)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.patch_dropout(x) if hasattr(v, "patch_dropout") else x
        x = v.ln_pre(x)
        # (B, S, D) -> (S, B, D)
        x = x.permute(1, 0, 2)

        for i, block in enumerate(v.transformer.resblocks):
            if i < self.Lv:
                # Insert / replace prompts at positions [1, 1+Pv) of the sequence.
                B = x.shape[1]
                prompts = self.vision_prompts[i].to(x.dtype).unsqueeze(1).expand(-1, B, -1)
                # x layout: [CLS, (old prompts or none), patches]
                if i == 0:
                    # Insert after cls
                    x = torch.cat([x[:1], prompts, x[1:]], dim=0)
                else:
                    # Replace previous prompt outputs in-place
                    x = torch.cat([x[:1], prompts, x[1 + self.Pv:]], dim=0)
            x = block(x)

        # After last layer, drop prompt positions if any remain.
        if self.Lv > 0:
            x = torch.cat([x[:1], x[1 + self.Pv:]], dim=0)
        x = x.permute(1, 0, 2)  # (B, S, D)
        x = v.ln_post(x[:, 0])  # cls token
        if v.proj is not None:
            x = x @ v.proj
        return F.normalize(x, dim=-1)

    # -------------------------------------------------------------------- text
    def tokenize_classes(self, class_names: Sequence[str], template: str = "a photo of a {}.") -> torch.Tensor:
        texts = [template.format(c.replace("_", " ")) for c in class_names]
        return self.tokenizer(texts)

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (C, n_ctx) -> (C, D) ℓ2-normalized embeddings."""
        m = self.model
        cast_dtype = m.transformer.get_cast_dtype()
        x = m.token_embedding(token_ids).to(cast_dtype)           # (C, n, D)
        x = x + m.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)                                    # (n, C, D)

        # Build attention mask of size (n + Pt) x (n + Pt) for ell >= 0.
        base_mask = m.attn_mask  # (n, n) upper-triangular -inf
        attn_mask_with = self._build_attn_mask_with_prompts(base_mask, self.Pt, cast_dtype, x.device)

        for i, block in enumerate(m.transformer.resblocks):
            if i < self.Lt and self.Pt > 0:
                C = x.shape[1]
                prompts = self.text_prompts[i].to(x.dtype).unsqueeze(1).expand(-1, C, -1)
                if i == 0:
                    x = torch.cat([x[:1], prompts, x[1:]], dim=0)
                else:
                    x = torch.cat([x[:1], prompts, x[1 + self.Pt:]], dim=0)
                block_mask = attn_mask_with
            else:
                block_mask = base_mask
            x = block(x, attn_mask=block_mask)

        if self.Lt > 0 and self.Pt > 0:
            x = torch.cat([x[:1], x[1 + self.Pt:]], dim=0)
        x = x.permute(1, 0, 2)                                    # (C, n, D)
        x = m.ln_final(x)

        # Pool at EOS (= argmax of token_ids over n dim).
        eos_idx = token_ids.argmax(dim=-1)                        # (C,)
        x = x[torch.arange(x.shape[0]), eos_idx]
        if m.text_projection is not None:
            if isinstance(m.text_projection, nn.Linear):
                x = m.text_projection(x)
            else:
                x = x @ m.text_projection
        return F.normalize(x, dim=-1)

    @staticmethod
    def _build_attn_mask_with_prompts(
        base_mask: torch.Tensor, Pt: int, dtype, device
    ) -> torch.Tensor:
        """Insert Pt prompt rows/cols at position 1 in a causal attn mask."""
        n = base_mask.shape[0]
        S = n + Pt
        m = torch.full((S, S), float("-inf"), dtype=base_mask.dtype, device=base_mask.device)
        # We treat prompts as living at positions [1, 1+Pt). Define a virtual
        # position map old_idx -> new_idx: 0 -> 0; k>=1 -> k + Pt.
        # Causal property: pos i attends to pos j iff j <= i.
        # SOS at 0 attends to itself only (and prompts are after it -> blocked).
        # Prompts at [1, 1+Pt) attend to SOS and themselves.
        # Original token at new position 1+Pt+k attends to everything up to itself.
        idx = torch.tril_indices(S, S)
        m[idx[0], idx[1]] = 0.0
        return m.to(dtype=dtype, device=device)

    # ------------------------------------------------------------------ utils
    def forward(
        self,
        images: torch.Tensor,
        class_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = self.encode_image(images)
        t = self.encode_text(class_token_ids)
        logits = self.logit_scale.exp() * v @ t.t()
        return v, t, logits

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        for p in self.parameters():
            if p.requires_grad:
                yield p


def build_student(cfg: StudentConfig) -> PromptCLIP:
    return PromptCLIP(cfg)
