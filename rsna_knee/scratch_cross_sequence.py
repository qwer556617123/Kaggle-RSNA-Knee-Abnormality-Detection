"""Scratch 2.5D MRI windows with controlled same-plane sequence exchange.

No external checkpoint, report text, voxel registration, or learned anatomy
location is used.  Control and candidate have identical parameters; only the
allowed attention edges differ.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ScratchCrossSequence(nn.Module):
    def __init__(self, *, cross_pair: bool, targets: int = 12, width: int = 128,
                 image_size: int = 160, heads: int = 4):
        super().__init__()
        if width % heads or image_size < 64:
            raise ValueError("invalid scratch model dimensions")
        self.cross_pair = bool(cross_pair)
        self.image_size = image_size
        self.heads = heads
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.GroupNorm(4, 24), nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, width, 3, stride=2, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.slot_embedding = nn.Embedding(5, width)
        self.depth_projection = nn.Linear(1, width)
        self.laterality_embedding = nn.Embedding(3, width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=.1, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(6 * width, width), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(width, targets))

    def attention_allowed(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """Return B,K,K edges; candidate adds only sagittal/coronal pairs."""
        if slot_ids.ndim != 2 or slot_ids.min() < 0 or slot_ids.max() > 4:
            raise ValueError("invalid sequence slot IDs")
        source = slot_ids[:, :, None]
        target = slot_ids[:, None, :]
        allowed = source == target
        if self.cross_pair:
            allowed = allowed | (((source == 1) & (target == 3)) |
                                 ((source == 3) & (target == 1)) |
                                 ((source == 2) & (target == 4)) |
                                 ((source == 4) & (target == 2)))
        return allowed

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
        if not valid.any(dim=1).all() or laterality.min() < 0 or laterality.max() > 2:
            raise ValueError("study lacks valid windows or has invalid laterality")
        x = images.flatten(0, 1)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear",
                              align_corners=False)
        tokens = self.backbone(x).reshape(batch, windows, -1)
        tokens = (tokens + self.slot_embedding(slot_ids.long())
                  + self.depth_projection(depth.unsqueeze(-1).to(tokens.dtype))
                  + self.laterality_embedding(laterality.long()).unsqueeze(1))
        # Per-batch masks support independently sampled 12-window layouts.
        allowed = self.attention_allowed(slot_ids) & valid[:, None, :]
        # Invalid queries retain one finite edge; final pooling removes them.
        diagonal = torch.eye(windows, dtype=torch.bool, device=images.device)[None]
        allowed = allowed | (diagonal & ~valid[:, :, None])
        disallowed = ~allowed
        disallowed = disallowed.unsqueeze(1).expand(batch, self.heads, windows, windows)
        disallowed = disallowed.reshape(batch * self.heads, windows, windows)
        mixed, _ = self.attention(tokens, tokens, tokens, attn_mask=disallowed,
                                  need_weights=False)
        tokens = self.norm(tokens + mixed)
        pooled = []
        for slot in range(5):
            selected = valid & slot_ids.eq(slot)
            pooled.append((tokens * selected.unsqueeze(-1)).sum(1) /
                          selected.sum(1, keepdim=True).clamp_min(1))
        global_pool = (tokens * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        return self.head(torch.cat((global_pool, *pooled), dim=-1))
