"""Scratch MRI lesion detector with a fixed spatial and slice MIL pooling rule.

The paired arms have identical parameters. Only the aggregation of the same
12 target logit maps differs. No report or external checkpoint is consumed.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ScratchSparseMIL(nn.Module):
    SPATIAL_K = 5
    WINDOW_K = 2

    def __init__(self, *, sparse_pool: bool, targets: int = 12,
                 width: int = 128, image_size: int = 160):
        super().__init__()
        if image_size < 64 or width % 8:
            raise ValueError("invalid scratch MIL dimensions")
        self.sparse_pool = bool(sparse_pool)
        self.image_size = image_size
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.GroupNorm(4, 24), nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, width, 3, stride=2, padding=1), nn.GroupNorm(8, width), nn.GELU(),
        )
        self.target_map = nn.Conv2d(width, targets, 1)
        self.slot_bias = nn.Embedding(5, targets)
        self.depth_bias = nn.Linear(1, targets, bias=False)
        self.laterality_bias = nn.Embedding(3, targets)

    def forward(self, images: torch.Tensor, *, window_mask: torch.Tensor,
                slot_ids: torch.Tensor, depth: torch.Tensor,
                laterality: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("expected B,K,3,H,W MRI windows")
        batch, windows = images.shape[:2]
        if (window_mask.shape != (batch, windows) or slot_ids.shape != (batch, windows)
                or depth.shape != (batch, windows) or laterality.shape != (batch,)):
            raise ValueError("MRI window metadata shape mismatch")
        valid = window_mask.bool()
        if (not valid.any(dim=1).all() or slot_ids.min() < 0 or slot_ids.max() > 4
                or laterality.min() < 0 or laterality.max() > 2
                or not torch.isfinite(depth).all()):
            raise ValueError("invalid study window metadata")
        x = images.flatten(0, 1)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        maps = self.target_map(self.backbone(x))
        maps = maps.reshape(batch, windows, maps.shape[1], -1)
        if self.sparse_pool:
            # Fixed, label-blind spatial support; no hyperparameter search.
            window_logits = maps.topk(min(self.SPATIAL_K, maps.shape[-1]), dim=-1).values.mean(-1)
        else:
            window_logits = maps.mean(-1)
        window_logits = (window_logits + self.slot_bias(slot_ids.long())
                         + self.depth_bias(depth.unsqueeze(-1).to(window_logits.dtype))
                         + self.laterality_bias(laterality.long()).unsqueeze(1))
        per_slot, present = [], []
        for slot in range(5):
            selected = valid & slot_ids.eq(slot)
            count = selected.sum(1, keepdim=True)
            present.append(count.gt(0))
            if self.sparse_pool:
                eligible = window_logits.masked_fill(~selected.unsqueeze(-1), float("-inf"))
                chosen = eligible.topk(min(self.WINDOW_K, windows), dim=1).values
                chosen = torch.where(torch.isfinite(chosen), chosen, torch.zeros_like(chosen))
                pooled = chosen.sum(1) / count.clamp(min=1, max=self.WINDOW_K)
            else:
                pooled = (window_logits * selected.unsqueeze(-1)).sum(1) / count.clamp_min(1)
            per_slot.append(pooled)
        slot_values = torch.stack(per_slot, dim=1)
        slot_present = torch.stack(present, dim=1)
        return (slot_values * slot_present).sum(1) / slot_present.sum(1).clamp_min(1)
