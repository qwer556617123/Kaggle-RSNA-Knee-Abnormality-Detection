"""S: hierarchical protocol pooling, leaving the historical control unchanged."""
import torch
from torch import nn
from .research_coatnet import SpatialEvidenceMIL


def build_series_model(root):
    """Preserve control initialization and RNG; add only the S query parameters."""
    from .research_coatnet import build_clean_spatial_model
    control = build_clean_spatial_model(root, pooling_arm='global')
    return from_control(control)


def from_control(control):
    if control.pooling_arm != 'global':
        raise ValueError('S requires the unchanged global control')
    targets, dimensions = control.class_weight.shape
    # Building the new head must not change the subsequent training RNG stream.
    with torch.random.fork_rng(devices=[]):
        model = SeriesEvidenceMIL(control.backbone, dimensions, targets)
        incompatible = model.load_state_dict(control.state_dict(), strict=False)
    if incompatible.missing_keys != ['slot_queries'] or incompatible.unexpected_keys:
        raise ValueError('unexpected S/control state difference')
    return model


class SeriesEvidenceMIL(SpatialEvidenceMIL):
    def __init__(self, backbone, dimensions, targets=12):
        super().__init__(backbone, dimensions, pooling_arm='global', targets=targets)
        self.slot_queries = nn.Parameter(torch.empty(targets, dimensions))
        nn.init.trunc_normal_(self.slot_queries, std=.02)

    def forward(self, images, window_mask=None, slot_ids=None, depth=None, laterality=None):
        if images.ndim != 5:
            raise ValueError('images must be B,K,C,H,W')
        batch, windows = images.shape[:2]
        if window_mask is None or window_mask.shape != (batch, windows):
            raise ValueError('window mask shape mismatch')
        if not window_mask.any(1).all():
            raise ValueError('study has no valid windows')
        if slot_ids is None or depth is None or laterality is None:
            raise ValueError('metadata required')
        if slot_ids.shape != (batch, windows) or depth.shape != (batch, windows) or laterality.shape != (batch,):
            raise ValueError('metadata shape mismatch')
        if slot_ids.min() < 0 or slot_ids.max() >= 5 or laterality.min() < 0 or laterality.max() >= 3:
            raise ValueError('metadata index out of range')
        if not torch.isfinite(depth).all() or depth.min() < 0 or depth.max() > 1:
            raise ValueError('invalid normalized depth')
        tokens, _, _ = self._tokens(images.flatten(0, 1))
        features = tokens.mean(1).reshape(batch, windows, -1)
        features = features + self.slot_embedding(slot_ids.long()) + self.depth_projection(depth.unsqueeze(-1).to(features.dtype)) + self.laterality_embedding(laterality.long()).unsqueeze(1)
        scores = self.attention(features)
        slot_tokens, slot_masks = [], []
        for slot in range(5):
            valid = window_mask.bool() & slot_ids.eq(slot)
            present = valid.any(1)
            logits = scores.masked_fill(~valid.unsqueeze(-1), -torch.inf)
            logits = torch.where(present[:, None, None], logits, torch.zeros_like(logits))
            weights = logits.softmax(1) * valid.unsqueeze(-1)
            slot_tokens.append(torch.einsum('bkt,bkd->btd', weights, features))
            slot_masks.append(present)
        tokens = torch.stack(slot_tokens, 1)
        mask = torch.stack(slot_masks, 1)
        scores = torch.einsum('bstd,td->bst', tokens, self.slot_queries) / features.shape[-1] ** .5
        weights = scores.masked_fill(~mask.unsqueeze(-1), -torch.inf).softmax(1)
        pooled = torch.einsum('bst,bstd->btd', weights, tokens)
        return (pooled * self.class_weight).sum(-1) + self.class_bias
