"""Equal-compute canonical-plane crop policies for the clean ROI study.

This module deliberately does *not* infer anatomy from reports or train a
localizer. It replaces a fixed subset of already decoded, canonical cache
windows with 96 mm crops, then rescales them back to the cache resolution.
The number of windows and backbone forwards is therefore unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .constants import SLOT_NAMES


CACHE_FOV_MM = 160.0
ROI_FOV_MM = 96.0
CACHE_PIXELS = 336
ROI_PIXELS = round(CACHE_PIXELS * ROI_FOV_MM / CACHE_FOV_MM)

_PLANE_OFFSETS = {
    0: ((-.18, 0.0), (.18, 0.0), (0.0, -.13), (0.0, .13)),
    1: ((-.18, 0.0), (.18, 0.0), (0.0, -.16), (0.0, .16)),
    2: ((-.18, 0.0), (.18, 0.0), (0.0, -.15), (0.0, .15)),
    3: ((-.18, 0.0), (.18, 0.0), (0.0, -.16), (0.0, .16)),
    4: ((-.18, 0.0), (.18, 0.0), (0.0, -.15), (0.0, .15)),
}
_SLOT_COUNTS = (4, 6, 6, 4, 4)


def local_window_mask(indices: np.ndarray) -> np.ndarray:
    """Return the fixed half-window replacement mask for cache indices."""
    values = np.asarray(indices, dtype=int)
    if values.ndim != 1 or ((values < 0) | (values >= sum(_SLOT_COUNTS))).any():
        raise ValueError('cache window indices must be in [0, 24)')
    starts = np.cumsum((0, *_SLOT_COUNTS[:-1]))
    within = values - np.asarray([starts[np.searchsorted(starts, value, side='right') - 1] for value in values])
    return (within % 2) == 1


def crop_offset(slot_id: int, window_index: int, arm: str) -> tuple[float, float]:
    """Select a pre-registered centre or plane-preserving ROI offset."""
    if arm == 'center':
        return 0.0, 0.0
    if arm != 'anatomic':
        raise ValueError('crop arm must be center or anatomic')
    if slot_id not in _PLANE_OFFSETS:
        raise ValueError('invalid protocol slot')
    return _PLANE_OFFSETS[slot_id][int(window_index) % len(_PLANE_OFFSETS[slot_id])]


def validate_window_metadata(metadata: list[dict]) -> None:
    """Fail closed if a ROI cache record lacks its canonical provenance."""
    if len(metadata) != sum(_SLOT_COUNTS):
        raise ValueError('ROI requires 24 window metadata records')
    expected = np.repeat(SLOT_NAMES, _SLOT_COUNTS)
    for index, (item, slot) in enumerate(zip(metadata, expected)):
        if item.get('slot') != slot:
            raise ValueError(f'ROI metadata slot mismatch at cache window {index}')
        detail = item.get('source_detail') or {}
        transform = detail.get('transform') or {}
        if item.get('missing', False):
            continue
        if not detail.get('pixel_spacing') or transform.get('fov_mm') != CACHE_FOV_MM:
            raise ValueError(f'ROI metadata has no approved physical transform at window {index}')
        if not transform.get('target_horizontal') or not transform.get('target_vertical'):
            raise ValueError(f'ROI metadata lacks canonical axes at window {index}')


def apply_roi_crops(images: np.ndarray, indices: np.ndarray, arm: str, *, crop_pixels: int = ROI_PIXELS) -> np.ndarray:
    """Replace a fixed half of K cache windows with same-shape 96 mm views.

    ``images`` is Kx3xHxW uint8 or float data. No flips, reordering, or
    target-specific pixel edits occur. Resampling changes only the field of
    view of marked views and always restores HxW, preserving encoder calls.
    """
    if arm not in {'center', 'anatomic'}:
        raise ValueError('ROI arm must be center or anatomic')
    values = np.asarray(images)
    selected = np.asarray(indices, dtype=int)
    if values.ndim != 4 or values.shape[0] != len(selected) or values.shape[1] != 3:
        raise ValueError('ROI image/index shape mismatch')
    height, width = values.shape[-2:]
    if height != width or crop_pixels <= 0 or crop_pixels > height:
        raise ValueError('ROI crop requires square cache images at least crop_pixels wide')
    result = values.copy()
    mask = local_window_mask(selected)
    starts = np.cumsum((0, *_SLOT_COUNTS[:-1]))
    for position, global_index in enumerate(selected):
        if not mask[position]:
            continue
        slot = int(np.searchsorted(starts, global_index, side='right') - 1)
        within = int(global_index - starts[slot])
        dx, dy = crop_offset(slot, within, arm)
        centre_x = round((.5 + dx) * (width - 1))
        centre_y = round((.5 + dy) * (height - 1))
        x0 = int(np.clip(centre_x - crop_pixels // 2, 0, width - crop_pixels))
        y0 = int(np.clip(centre_y - crop_pixels // 2, 0, height - crop_pixels))
        patch = torch.from_numpy(result[position:position + 1, :, y0:y0 + crop_pixels, x0:x0 + crop_pixels].astype(np.float32))
        patch = F.interpolate(patch, size=(height, width), mode='bilinear', align_corners=False)[0]
        if np.issubdtype(result.dtype, np.integer):
            result[position] = patch.round().clamp(0, np.iinfo(result.dtype).max).to(torch.uint8).numpy()
        else:
            result[position] = patch.numpy().astype(result.dtype, copy=False)
    return result
