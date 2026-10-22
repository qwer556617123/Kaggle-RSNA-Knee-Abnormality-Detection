from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import LABELS, SLOT_NAMES


@dataclass(frozen=True)
class FeatureHeadConfig:
    input_dimension: int = 1024
    hidden_dimension: int = 192
    slots: tuple[str, ...] = SLOT_NAMES
    dropout: float = 0.20
    clinical_prior_strength: float = 0.18
    architecture_version: str = "target-query-v1"


def _clinical_slot_prior(slots: tuple[str, ...]) -> np.ndarray:
    prior = np.zeros((len(LABELS), len(slots)), dtype=np.float32)
    for target_index, target in enumerate(LABELS):
        for slot_index, slot_name in enumerate(slots):
            slot = slot_name.casefold()
            sagittal, coronal, axial = (name in slot for name in ("sagittal", "coronal", "axial"))
            fluid = "fluid" in slot and "nonfluid" not in slot
            structural = "nonfluid" in slot
            if target == "ACL":
                score = 1.0 * sagittal + 0.25 * axial
            elif target == "MCL":
                score = 1.0 * coronal + 0.20 * sagittal
            elif "Meniscus" in target:
                score = 0.85 * coronal + 0.75 * sagittal
            elif target in ("Medial OA", "Lateral OA"):
                score = 0.85 * coronal + 0.35 * sagittal + 0.25 * structural
            elif target == "PF OA":
                score = 1.0 * axial + 0.35 * sagittal
            elif target in ("Effusion", "Synovitis"):
                score = 0.9 * axial + 0.65 * sagittal + 0.35 * coronal + 0.25 * fluid
            elif target == "Baker's":
                score = 0.95 * sagittal + 0.45 * axial
            elif target == "Contusion":
                score = 0.55 + 0.35 * fluid
            else:
                score = 0.45 + 0.25 * structural + 0.20 * fluid
            prior[target_index, slot_index] = score
    prior -= prior.mean(axis=1, keepdims=True)
    prior /= np.maximum(prior.std(axis=1, keepdims=True), 1e-6)
    return prior


def create_feature_attention_head(config: FeatureHeadConfig):
    """AnyMC3D-style target-query slice and protocol attention head."""
    import torch
    from torch import nn

    class FeatureAttentionHead(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = config.hidden_dimension
            self.projection = nn.Sequential(
                nn.LayerNorm(config.input_dimension),
                nn.Linear(config.input_dimension, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.slot_embedding = nn.Parameter(torch.zeros(len(config.slots), hidden))
            self.slice_queries = nn.Parameter(torch.zeros(len(LABELS), hidden))
            self.slot_queries = nn.Parameter(torch.zeros(len(LABELS), hidden))
            self.target_norm = nn.LayerNorm(hidden)
            self.target_weight = nn.Parameter(torch.empty(len(LABELS), hidden))
            self.target_bias = nn.Parameter(torch.zeros(len(LABELS)))
            self.register_buffer(
                "clinical_slot_prior",
                torch.from_numpy(_clinical_slot_prior(config.slots)),
            )
            nn.init.normal_(self.slot_embedding, std=0.02)
            nn.init.normal_(self.slice_queries, std=0.02)
            nn.init.normal_(self.slot_queries, std=0.02)
            nn.init.trunc_normal_(self.target_weight, std=0.02)

        @staticmethod
        def masked_softmax(logits, mask, dim):
            probabilities = logits.masked_fill(~mask.bool(), -1e4).softmax(dim=dim)
            probabilities = probabilities * mask.to(logits.dtype)
            return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(1e-6)

        def forward(self, features, slot_mask, slice_mask=None):
            # features [B, protocol slots, native slices, D]
            projected = self.projection(features.float())
            projected = projected + self.slot_embedding[None, :, None, :]
            scale = projected.shape[-1] ** 0.5
            slice_logits = torch.einsum("bvsd,td->btvs", projected, self.slice_queries) / scale
            if slice_mask is None:
                slice_mask = slot_mask[:, :, None].expand(-1, -1, features.shape[2])
            slice_valid = slice_mask[:, None, :, :].bool().expand_as(slice_logits)
            slice_weights = self.masked_softmax(slice_logits, slice_valid, dim=-1)
            slot_features = torch.einsum("btvs,bvsd->btvd", slice_weights, projected)
            slot_logits = torch.einsum("btvd,td->btv", slot_features, self.slot_queries) / scale
            slot_logits = slot_logits + config.clinical_prior_strength * self.clinical_slot_prior[None]
            slot_weights = self.masked_softmax(slot_logits, slot_mask[:, None, :].bool(), dim=-1)
            target_features = torch.einsum("btv,btvd->btd", slot_weights, slot_features)
            target_features = self.target_norm(target_features)
            return (target_features * self.target_weight[None]).sum(-1) + self.target_bias

    class GatedMILHead(nn.Module):
        """Target-wise gated attention with a sparse-lesion max branch."""

        def __init__(self):
            super().__init__()
            hidden = config.hidden_dimension
            self.projection = nn.Sequential(
                nn.LayerNorm(config.input_dimension),
                nn.Linear(config.input_dimension, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.slot_embedding = nn.Parameter(torch.zeros(len(config.slots), hidden))
            self.attention_v = nn.Linear(hidden, hidden)
            self.attention_u = nn.Linear(hidden, hidden)
            self.attention_out = nn.Linear(hidden, len(LABELS), bias=False)
            self.target_norm = nn.LayerNorm(hidden)
            self.target_weight = nn.Parameter(torch.empty(len(LABELS), hidden))
            self.target_bias = nn.Parameter(torch.zeros(len(LABELS)))
            self.max_weight = nn.Parameter(torch.empty(len(LABELS), hidden))
            self.max_bias = nn.Parameter(torch.zeros(len(LABELS)))
            self.mix_logit = nn.Parameter(torch.full((len(LABELS),), -1.1))
            self.register_buffer(
                "clinical_slot_prior",
                torch.from_numpy(_clinical_slot_prior(config.slots)),
            )
            nn.init.normal_(self.slot_embedding, std=0.02)
            nn.init.trunc_normal_(self.target_weight, std=0.02)
            nn.init.trunc_normal_(self.max_weight, std=0.02)

        def forward(self, features, slot_mask, slice_mask=None):
            projected = self.projection(features.float())
            projected = projected + self.slot_embedding[None, :, None, :]
            if slice_mask is None:
                slice_mask = slot_mask[:, :, None].expand(-1, -1, features.shape[2])
            valid = slice_mask.bool()
            gate = torch.tanh(self.attention_v(projected)) * torch.sigmoid(self.attention_u(projected))
            logits = self.attention_out(gate).permute(0, 3, 1, 2)
            logits = logits + config.clinical_prior_strength * self.clinical_slot_prior[None, :, :, None]
            logits = logits.masked_fill(~valid[:, None], -1e4)
            weights = logits.flatten(2).softmax(dim=-1).reshape_as(logits)
            weights = weights * valid[:, None].to(weights.dtype)
            weights = weights / weights.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            pooled = torch.einsum("btvs,bvsd->btd", weights, projected)
            pooled = self.target_norm(pooled)
            attention_logits = (pooled * self.target_weight[None]).sum(-1) + self.target_bias

            token_logits = torch.einsum("bvsd,td->btvs", projected, self.max_weight)
            token_logits = token_logits + self.max_bias[None, :, None, None]
            max_logits = token_logits.masked_fill(~valid[:, None], -1e4).flatten(2).max(dim=-1).values
            mix = self.mix_logit.sigmoid()[None]
            return (1.0 - mix) * attention_logits + mix * max_logits

    if config.architecture_version == "target-query-v1":
        return FeatureAttentionHead()
    if config.architecture_version == "gated-max-mil-v2":
        return GatedMILHead()
    raise ValueError(f"unsupported feature-head architecture {config.architecture_version!r}")
