"""Fold-local masked-window feature learning for knee MRI studies."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MaskedSliceEncoder(nn.Module):
    """Predict held-out neighboring-window features with a fixed 25% mask."""
    def __init__(self, feature_dim: int, *, hidden_dim: int = 256, targets: int = 12):
        super().__init__()
        if feature_dim < 1 or hidden_dim % 4:
            raise ValueError('feature dimensions must be positive and hidden_dim divisible by four')
        self.project = nn.Linear(feature_dim, hidden_dim)
        self.slot_embedding = nn.Embedding(5, hidden_dim)
        self.depth_embedding = nn.Linear(1, hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, 4, dim_feedforward=hidden_dim * 2,
                                           dropout=.1, batch_first=True, norm_first=True)
        self.sequence = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.reconstruct = nn.Linear(hidden_dim, feature_dim)
        self.target_head = nn.Linear(hidden_dim, targets)

    @staticmethod
    def block_mask(valid: torch.Tensor, slots: torch.Tensor, *, seed: int = 20260905) -> torch.Tensor:
        if valid.shape != slots.shape or valid.ndim != 2:
            raise ValueError('valid and slot masks must align as B,K')
        generator = torch.Generator(device=valid.device).manual_seed(seed)
        selected = torch.zeros_like(valid, dtype=torch.bool)
        for batch in range(valid.shape[0]):
            for slot in range(5):
                positions = torch.where(valid[batch] & (slots[batch] == slot))[0]
                if len(positions) < 2:
                    continue
                span = max(1, round(.25 * len(positions)))
                start = int(torch.randint(len(positions) - span + 1, (1,), generator=generator,
                                          device=valid.device))
                selected[batch, positions[start:start + span]] = True
        if not selected.any():
            raise ValueError('study has no maskable slice group')
        return selected

    def encode(self, features, valid, slots, depth, *, masked=None):
        if features.ndim != 3 or valid.shape != features.shape[:2] or slots.shape != valid.shape or depth.shape != valid.shape:
            raise ValueError('feature and MRI sequence metadata shapes differ')
        if not valid.any(dim=1).all():
            raise ValueError('study has no valid windows')
        x = self.project(features) + self.slot_embedding(slots.long()) + self.depth_embedding(depth.unsqueeze(-1))
        if masked is not None:
            if masked.shape != valid.shape or (masked & ~valid).any():
                raise ValueError('masked windows must be valid')
            x = torch.where(masked.unsqueeze(-1), self.mask_token.view(1, 1, -1), x)
        return self.sequence(x, src_key_padding_mask=~valid)

    def reconstruction_loss(self, features, valid, slots, depth, *, seed=20260905):
        masked = self.block_mask(valid, slots, seed=seed)
        hidden = self.encode(features, valid, slots, depth, masked=masked)
        predicted = self.reconstruct(hidden[masked])
        target = features[masked].detach()
        loss = (1 - F.cosine_similarity(predicted, target, dim=-1)).mean()
        return loss, {'masked_windows': int(masked.sum()), 'mask_fraction': float(masked.sum() / valid.sum())}

    def forward(self, features, valid, slots, depth):
        hidden = self.encode(features, valid, slots, depth)
        pooled = (hidden * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True)
        return self.target_head(pooled)
