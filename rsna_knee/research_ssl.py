"""Competition-MRI-only Barlow Twins initialization for a scratch study encoder.

No report labels, pretrained weights, or test MRI enter this pretraining path.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rsna_knee.scratch_sparse_mil import ScratchSparseMIL


class MRIBarlowEncoder(nn.Module):
    def __init__(self, *, image_size: int = 160, width: int = 128, projection_dim: int = 128):
        super().__init__()
        self.backbone = ScratchSparseMIL(sparse_pool=True, image_size=image_size, width=width).backbone
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(width, width * 2), nn.BatchNorm1d(width * 2), nn.GELU(),
            nn.Linear(width * 2, projection_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("expected N,3,H,W MRI windows")
        return self.projector(self.backbone(images))


def augment_mri_windows(images: torch.Tensor) -> torch.Tensor:
    """Mild stochastic geometry and intensity transforms on normalized MRI windows."""
    if images.ndim != 4 or images.shape[1] != 3 or not torch.isfinite(images).all():
        raise ValueError("invalid MRI image batch")
    count = images.shape[0]
    if count < 2:
        raise ValueError("at least two valid MRI windows required")
    device, dtype = images.device, images.dtype
    scale = 1.0 + (torch.rand(count, device=device) - 0.5) * 0.16
    shift = (torch.rand(count, 2, device=device) - 0.5) * 0.08
    flip = torch.where(torch.rand(count, device=device) < 0.5, -1.0, 1.0)
    theta = torch.zeros(count, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = (scale * flip).to(dtype)
    theta[:, 1, 1] = scale.to(dtype)
    theta[:, :, 2] = shift.to(dtype)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    transformed = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)
    gain = 1.0 + (torch.rand(count, 1, 1, 1, device=device) - 0.5) * 0.2
    bias = (torch.rand(count, 1, 1, 1, device=device) - 0.5) * 0.08
    noise = torch.randn_like(transformed) * 0.02
    return (transformed * gain + bias + noise).clamp(-1.0, 1.0)


def barlow_loss(first: torch.Tensor, second: torch.Tensor, *, offdiag_weight: float = 0.005) -> torch.Tensor:
    if first.ndim != 2 or first.shape != second.shape or first.shape[0] < 2:
        raise ValueError("paired N,D SSL embeddings required")
    a, b = first.float(), second.float()
    a = (a - a.mean(0)) / a.std(0, unbiased=False).clamp_min(1e-4)
    b = (b - b.mean(0)) / b.std(0, unbiased=False).clamp_min(1e-4)
    corr = a.T @ b / a.shape[0]
    diag = torch.diagonal(corr)
    off = corr - torch.diag_embed(diag)
    return (diag - 1).square().sum() + offdiag_weight * off.square().sum()
