"""Learned local evidence reader for study-level MRI classification.

The module predicts target-specific sampling regions from low-resolution windows,
then re-encodes those regions from the supplied high-resolution image tensor.
It never claims the regions are anatomical segmentations.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .constants import LABELS


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1)


def _encode(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    value = encoder(images)
    if isinstance(value, (tuple, list)):
        value = value[0]
    if value.ndim > 2:
        value = value.mean(tuple(range(2, value.ndim)))
    if value.ndim != 2:
        raise ValueError("encoder must return one feature vector per image")
    return value


class LearnedRegionMIL(nn.Module):
    """Target-specific differentiable crop reader with continuity diagnostics."""
    def __init__(self, global_encoder: nn.Module, local_encoder: nn.Module, dimension: int,
                 *, targets: int = len(LABELS), regions_per_target: int = 2,
                 crop_fraction: float = .35, local_size: int = 384, global_size: int = 384,
                 local_batch_size: int = 2):
        super().__init__()
        if not 0 < crop_fraction <= 1 or regions_per_target < 1:
            raise ValueError("invalid region reader configuration")
        self.global_encoder, self.local_encoder = global_encoder, local_encoder
        self.targets, self.regions_per_target = targets, regions_per_target
        if local_batch_size < 1:
            raise ValueError("local_batch_size must be positive")
        self.crop_fraction, self.local_size, self.global_size = crop_fraction, local_size, global_size
        self.local_batch_size = local_batch_size
        self.locator = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, targets * regions_per_target * 2))
        self.target_queries = nn.Parameter(torch.randn(targets, dimension) * .02)
        self.head = nn.Sequential(nn.LayerNorm(dimension * 2), nn.Dropout(.2), nn.Linear(dimension * 2, targets))

    def _grid(self, centers: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, targets, regions, _ = centers.shape
        axis = torch.linspace(-1, 1, self.local_size, dtype=centers.dtype, device=centers.device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        delta = torch.stack((xx, yy), dim=-1) * self.crop_fraction
        grid = centers[..., None, None, :] + delta
        return grid.clamp(-1, 1).reshape(batch * targets * regions, self.local_size, self.local_size, 2)

    def _crop_and_encode(self, highres: torch.Tensor, centers: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, windows, channels, height, width = highres.shape
        global_image = _masked_mean(highres.flatten(2), valid).reshape(batch, channels, height, width)
        repeated = global_image[:, None, None].expand(-1, self.targets, self.regions_per_target, -1, -1, -1)
        crops = F.grid_sample(repeated.reshape(-1, channels, height, width), self._grid(centers, height, width),
                              mode="bilinear", padding_mode="border", align_corners=False)
        encoded = torch.cat([_encode(self.local_encoder, crops[start:start + self.local_batch_size])
                             for start in range(0, len(crops), self.local_batch_size)], dim=0)
        return encoded.reshape(batch, self.targets, self.regions_per_target, -1).mean(2)

    def forward(self, images: torch.Tensor, *, window_mask: torch.Tensor,
                highres_images: torch.Tensor | None = None, return_diagnostics: bool = False):
        if images.ndim != 5 or window_mask.shape != images.shape[:2]:
            raise ValueError("images must be B,K,C,H,W with a matching window_mask")
        source = images if highres_images is None else highres_images
        if source.ndim != 5 or source.shape[:2] != images.shape[:2]:
            raise ValueError("highres_images must align with low-resolution windows")
        batch, windows = images.shape[:2]
        lowres = F.interpolate(images.flatten(0, 1), size=(self.global_size, self.global_size), mode="bilinear", align_corners=False)
        global_features = _encode(self.global_encoder, lowres).reshape(batch, windows, -1)
        pooled = _masked_mean(global_features, window_mask)
        centers = self.locator(pooled).reshape(batch, self.targets, self.regions_per_target, 2).tanh()
        local = self._crop_and_encode(source, centers, window_mask)
        global_by_target = pooled[:, None, :].expand(-1, self.targets, -1)
        logits = self.head(torch.cat((global_by_target, local), dim=-1)).diagonal(dim1=1, dim2=2)
        if not return_diagnostics:
            return logits
        continuity = (centers[:, :, 1:] - centers[:, :, :-1]).square().mean() if self.regions_per_target > 1 else centers.new_zeros(())
        return logits, {"region_centers": centers, "continuity_loss": continuity,
                        "local_ablation_logits": self.head(torch.cat((global_by_target, torch.zeros_like(local)), dim=-1)).diagonal(dim1=1, dim2=2)}

def build_clean_local_reader(root, *, regions_per_target: int = 2, global_size: int = 384, local_size: int = 384):
    """Build the local reader from the same pinned ImageNet initialization."""
    import json
    from pathlib import Path
    import timm
    from safetensors.torch import load_file
    from .research_coatnet import ARCH, PRETRAIN_SHA, file_sha256, strict_encoder_load
    root = Path(root)
    checkpoint = root / "model.safetensors"
    manifest = json.loads((root / "pretrain_manifest.json").read_text())
    if file_sha256(checkpoint) != PRETRAIN_SHA or manifest.get("sha256") != PRETRAIN_SHA or manifest.get("competition_supervision") is not False:
        raise ValueError("not the approved clean ImageNet checkpoint")
    backbone = timm.create_model(ARCH, pretrained=False, num_classes=0, global_pool="")
    strict_encoder_load(backbone, load_file(str(checkpoint)))
    for name, parameter in backbone.named_parameters():
        parameter.requires_grad_(name.startswith(("stages.2.", "stages.3.", "head.")))
    return LearnedRegionMIL(backbone, backbone, int(backbone.num_features), regions_per_target=regions_per_target,
                            global_size=global_size, local_size=local_size)
