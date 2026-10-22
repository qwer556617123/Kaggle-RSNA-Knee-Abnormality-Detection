"""Clean ImageNet initialization and strict contracts for matched research runs.

The public 44-slice corpus is allowed for compute profiling only until its
series-boundary provenance is established. Competition-finetuned weights are
never accepted as a clean OOF initialization.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .constants import LABELS, CONFIDENCE_COLUMNS
from .research_validation import UID, table_digest, nested_partition

ARCH = 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
PRETRAIN_SHA = '91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def load_contract(root, arm='control'):
    if arm not in ('control', 'evidence'):
        raise ValueError('unknown supervision arm')
    root = Path(root)
    manifest = json.loads((root / 'research_manifest.json').read_text())
    if manifest['contract_version'] != 'top10-clean-v1':
        raise ValueError('unsupported research contract')
    evaluation = pd.read_csv(root / 'evaluation.csv', dtype={UID: str})
    control = pd.read_csv(root / 'labels_control.csv', dtype={UID: str})
    candidate = pd.read_csv(root / 'labels_evidence.csv', dtype={UID: str})
    for frame, key in [(evaluation, 'evaluation_sha256'), (control, 'label_control_sha256'), (candidate, 'label_evidence_sha256')]:
        if table_digest(frame) != manifest[key]:
            raise ValueError(f'contract hash mismatch: {key}')
        if frame[UID].duplicated().any() or not frame[UID].equals(evaluation[UID]):
            raise ValueError('contract UID order mismatch')
    if table_digest(evaluation[[UID, 'report_group', 'fold']]) != manifest['fold_sha256']:
        raise ValueError('fold contract mismatch')
    if not np.array_equal(control[LABELS], candidate[LABELS]):
        raise ValueError('confidence-only experiment changed targets')
    for frame in (control, candidate):
        values = frame[LABELS + CONFIDENCE_COLUMNS].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('invalid targets or confidence')
        if not frame.report_group.equals(evaluation.report_group):
            raise ValueError('label report group mismatch')
    for fold in range(3):
        split = nested_partition(evaluation, fold)
        for name in ('train', 'inner', 'outer', 'excluded'):
            uids = evaluation.iloc[getattr(split, name)][UID]
            receipt = manifest['nested_splits'][str(fold)][name]
            digest = hashlib.sha256('\n'.join(uids).encode()).hexdigest()
            if len(uids) != receipt['count'] or digest != receipt['uid_sha256']:
                raise ValueError('nested partition differs from frozen manifest')
    return evaluation, control if arm == 'control' else candidate, manifest


def strict_encoder_load(backbone, state):
    """Only the declared ImageNet classifier may differ; even head.norm matters."""
    expected = backbone.state_dict()
    allowed = {'head.fc.weight', 'head.fc.bias'}
    removable = allowed - set(expected)
    filtered = {k: v for k, v in state.items() if k not in removable}
    if set(filtered) != set(expected):
        raise ValueError(f'encoder key mismatch: missing={sorted(set(expected)-set(filtered))}; unexpected={sorted(set(filtered)-set(expected))}')
    for key in expected:
        if expected[key].shape != filtered[key].shape:
            raise ValueError(f'encoder shape mismatch: {key}')
    backbone.load_state_dict(filtered, strict=True)


class CleanCoAtNetMIL(nn.Module):
    def __init__(self, backbone, dimensions, targets=12):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(dimensions)
        self.attention = nn.Sequential(nn.Linear(dimensions, 256), nn.Tanh(), nn.Dropout(.2), nn.Linear(256, targets))
        self.class_weight = nn.Parameter(torch.empty(targets, dimensions))
        self.class_bias = nn.Parameter(torch.zeros(targets))
        nn.init.trunc_normal_(self.class_weight, std=.02)

    def forward(self, images):
        batch, windows = images.shape[:2]
        features = self.backbone(images.flatten(0, 1)).reshape(batch, windows, -1)
        features = self.norm(features)
        weights = self.attention(features).softmax(dim=1)
        pooled = torch.einsum('bkt,bkf->btf', weights, features)
        return (pooled * self.class_weight).sum(-1) + self.class_bias


class SpatialEvidenceMIL(nn.Module):
    """Global control / fixed-center / target-query spatial evidence heads.

    All arms call the same backbone exactly once per window. This is a feature
    pooling ablation, not a segmentation or anatomical-localizer claim.
    """
    def __init__(self, backbone, dimensions, pooling_arm='global', targets=12):
        super().__init__()
        if pooling_arm not in ('global', 'center', 'target_query'):
            raise ValueError('unknown pooling arm')
        self.backbone, self.pooling_arm = backbone, pooling_arm
        self.norm = nn.LayerNorm(dimensions)
        self.attention = nn.Sequential(nn.Linear(dimensions, 256), nn.Tanh(), nn.Dropout(.2), nn.Linear(256, targets))
        self.class_weight = nn.Parameter(torch.empty(targets, dimensions))
        self.class_bias = nn.Parameter(torch.zeros(targets))
        self.local_queries = nn.Parameter(torch.empty(targets, dimensions))
        self.slot_embedding = nn.Embedding(5, dimensions)
        self.depth_projection = nn.Sequential(nn.Linear(1, dimensions), nn.Tanh())
        self.laterality_embedding = nn.Embedding(3, dimensions)
        # local mixing = .5 * sigmoid(parameter), initialized at 0.1.
        self.local_mix_logit = nn.Parameter(torch.full((targets,), float(np.log(.2/.8))))
        nn.init.trunc_normal_(self.class_weight, std=.02)
        nn.init.trunc_normal_(self.local_queries, std=.02)

    def _tokens(self, images):
        features = self.backbone.forward_features(images) if hasattr(self.backbone, 'forward_features') else self.backbone(images)
        if features.ndim == 4:
            batch, channels, height, width = features.shape
            tokens = features.flatten(2).transpose(1, 2)
        elif features.ndim == 3:
            batch, token_count, channels = features.shape
            side = int(np.sqrt(token_count))
            if side * side != token_count:
                raise ValueError('spatial evidence needs square feature tokens')
            height = width = side
            tokens = features
        else:
            raise ValueError('backbone must expose a spatial feature map')
        if tokens.shape[-1] != self.class_weight.shape[-1]:
            raise ValueError('spatial feature dimension mismatch')
        return self.norm(tokens), height, width

    def _pool(self, tokens, height, width):
        global_token = tokens.mean(dim=1)
        if self.pooling_arm == 'global':
            return global_token, None
        if height < 4 or width < 4:
            raise ValueError('4x4 regional pooling needs at least 4x4 features')
        spatial = tokens.reshape(tokens.shape[0], height, width, -1)
        row_edges = np.linspace(0, height, 5).round().astype(int)
        col_edges = np.linspace(0, width, 5).round().astype(int)
        regions = []
        for row in range(4):
            for col in range(4):
                regions.append(spatial[:, row_edges[row]:row_edges[row+1], col_edges[col]:col_edges[col+1]].mean((1,2)))
        regional = torch.stack(regions, dim=1)
        if self.pooling_arm == 'center':
            local = regional[:, [5,6,9,10]].mean(1)
        else:
            local_scores = torch.einsum('td,bkd->btk', self.local_queries, regional)
            local = torch.einsum('btk,bkd->btd', local_scores.softmax(-1), regional)
        return global_token, local

    def forward(self, images, window_mask=None, slot_ids=None, depth=None, laterality=None, return_features=False):
        if images.ndim != 5:
            raise ValueError('images must be B,K,C,H,W')
        batch, windows = images.shape[:2]
        tokens, height, width = self._tokens(images.flatten(0, 1))
        global_token, local = self._pool(tokens, height, width)
        global_token = global_token.reshape(batch, windows, -1)
        if local is not None:
            local = local.reshape(batch, windows, *local.shape[1:])
        if window_mask is None:
            window_mask = torch.ones((batch, windows), dtype=torch.bool, device=images.device)
        if window_mask.shape != (batch, windows):
            raise ValueError('window mask shape mismatch')
        if not window_mask.any(dim=1).all():
            raise ValueError('study has no valid windows')
        if slot_ids is None or depth is None or laterality is None:
            raise ValueError('slot, depth and laterality metadata are required')
        if slot_ids.shape != (batch, windows) or depth.shape != (batch, windows) or laterality.shape != (batch,):
            raise ValueError('metadata shape mismatch')
        if slot_ids.min() < 0 or slot_ids.max() >= 5 or laterality.min() < 0 or laterality.max() >= 3:
            raise ValueError('metadata index out of range')
        if not torch.isfinite(depth).all() or depth.min() < 0 or depth.max() > 1:
            raise ValueError('invalid normalized depth')
        metadata = (self.slot_embedding(slot_ids.long())
                    + self.depth_projection(depth.unsqueeze(-1).to(global_token.dtype))
                    + self.laterality_embedding(laterality.long()).unsqueeze(1))
        global_token = global_token + metadata
        scores = self.attention(global_token)
        scores = scores.masked_fill(~window_mask.unsqueeze(-1), -torch.inf)
        weights = scores.softmax(dim=1)
        pooled_global = torch.einsum('bkt,bkd->btd', weights, global_token)
        if local is not None:
            if local.ndim == 3:  # fixed center vector shared by all targets
                pooled_local = torch.einsum('bkt,bkd->btd', weights, local)
            else:  # target-query vectors
                pooled_local = torch.einsum('bkt,bktd->btd', weights, local)
            mix = .5 * torch.sigmoid(self.local_mix_logit).view(1,-1,1)
            pooled = pooled_global * (1-mix) + pooled_local * mix
        else:
            pooled = pooled_global
        logits = (pooled * self.class_weight.view(1,*self.class_weight.shape)).sum(-1) + self.class_bias.view(1,-1)
        return (logits, pooled) if return_features else logits


def build_clean_spatial_model(root, pooling_arm='global'):
    """Build a spatial model with the same strict ImageNet checkpoint contract."""
    import timm
    from safetensors.torch import load_file
    root = Path(root)
    path = root / 'model.safetensors'
    manifest = json.loads((root / 'pretrain_manifest.json').read_text())
    if file_sha256(path) != PRETRAIN_SHA or manifest['sha256'] != PRETRAIN_SHA or manifest['competition_supervision'] is not False:
        raise ValueError('not the approved clean ImageNet checkpoint')
    backbone = timm.create_model(ARCH, pretrained=False, num_classes=0, global_pool='')
    strict_encoder_load(backbone, load_file(str(path)))
    dimensions = getattr(backbone, 'num_features', None)
    if not dimensions:
        raise ValueError('spatial encoder has no num_features')
    model = SpatialEvidenceMIL(backbone, dimensions, pooling_arm=pooling_arm)
    for name, parameter in backbone.named_parameters():
        parameter.requires_grad_(name.startswith(('stages.2.', 'stages.3.', 'head.')))
    return model


def build_clean_model(root):
    import timm
    from safetensors.torch import load_file
    root = Path(root)
    path = root / 'model.safetensors'
    manifest = json.loads((root / 'pretrain_manifest.json').read_text())
    if file_sha256(path) != PRETRAIN_SHA or manifest['sha256'] != PRETRAIN_SHA or manifest['competition_supervision'] is not False:
        raise ValueError('not the approved clean ImageNet checkpoint')
    config = json.loads((root / 'config.json').read_text())
    if config['architecture'] != ARCH.split('.')[0] or config['pretrained_cfg']['input_size'] != [3,384,384]:
        raise ValueError('unexpected checkpoint architecture/input size')
    if config['pretrained_cfg']['mean'] != [.5]*3 or config['pretrained_cfg']['std'] != [.5]*3:
        raise ValueError('unexpected pretrained normalization')
    backbone = timm.create_model(ARCH, pretrained=False, num_classes=0, global_pool='avg')
    strict_encoder_load(backbone, load_file(str(path)))
    model = CleanCoAtNetMIL(backbone, backbone.num_features)
    # Budgeted adaptation: train the final two stages and MIL head; no label-based
    # layer selection. Freeze BatchNorm statistics in the fixed feature extractor.
    for name, parameter in backbone.named_parameters():
        parameter.requires_grad_(name.startswith(('stages.2.', 'stages.3.', 'head.')))
    return model


def freeze_batchnorm_statistics(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def soft_weighted_bce(logits, targets, confidence, loss_mask=None):
    if logits.shape != targets.shape or targets.shape != confidence.shape:
        raise ValueError('loss shape mismatch')
    if not torch.isfinite(confidence).all() or torch.any(confidence < 0) or torch.any(confidence > 1):
        raise ValueError('invalid loss confidence')
    if loss_mask is None:
        loss_mask = torch.ones_like(confidence)
    if loss_mask.shape != confidence.shape or not torch.isfinite(loss_mask).all() or torch.any(loss_mask < 0) or torch.any(loss_mask > 1):
        raise ValueError('invalid loss mask')
    # The denominator remains B*T, rather than active labels: lower-evidence
    # studies therefore contribute less gradient and cannot be upweighted by
    # masking. Batch=1 does not support pairwise AUC.
    return (nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none') * confidence * loss_mask).mean()
