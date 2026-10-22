"""Report-free, provenance-bound test inference for masked-slice candidates."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .constants import LABELS, SLOT_NAMES
from .dicom import slot_for_series
from .masked_slice import MaskedSliceEncoder
from .research_cache import ResearchCacheConfig, build_study_cache

COUNTS = (4, 6, 6, 4, 4)


def prepare_series_table(series: pd.DataFrame) -> pd.DataFrame:
    required = {'StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane', 'Fluid_Sensitive'}
    if not required <= set(series):
        raise ValueError(f'test series table lacks {sorted(required-set(series))}')
    result = series.copy()
    result['StudyInstanceUID'] = result['StudyInstanceUID'].astype(str)
    result['SeriesInstanceUID'] = result['SeriesInstanceUID'].astype(str)
    result['slot'] = [slot_for_series(plane, fluid)
                      for plane, fluid in zip(result.Anatomical_Plane, result.Fluid_Sensitive)]
    return result[result.slot.isin(SLOT_NAMES)].copy()


def study_visual_features(visual, study_uid: str, series: pd.DataFrame, dicom_root: str | Path,
                          *, device: torch.device, chunk_size: int = 4):
    """Match the clean training cache, 336-to-384 resize, and fixed 24 slots."""
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    started = time.perf_counter()
    pixels, masks, metadata, failures = build_study_cache(
        study_uid, series, dicom_root, ResearchCacheConfig())
    if pixels.shape != (24, 3, 336, 336) or masks.shape != (24,):
        raise RuntimeError('test MRI violated the fixed 24-window cache contract')
    valid = masks > .5
    if not valid.any():
        raise RuntimeError('test study has no valid MRI windows')
    slots = np.repeat(np.arange(5, dtype=np.int64), COUNTS)
    depths = np.concatenate([np.linspace(0., 1., count, dtype=np.float32) for count in COUNTS])
    x = torch.from_numpy(pixels.astype(np.float32) / 255.).sub(.5).div(.5)
    features = np.zeros((24, int(visual.num_features)), dtype=np.float32)
    indexes = np.flatnonzero(valid)
    visual.eval()
    with torch.inference_mode():
        for selected in np.array_split(indexes, max(1, int(np.ceil(len(indexes)/chunk_size)))):
            batch = F.interpolate(x[selected].to(device), size=(384, 384),
                                  mode='bilinear', align_corners=False)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                encoded = visual(batch)
            features[selected] = encoded.float().cpu().numpy()
    if not np.isfinite(features).all():
        raise RuntimeError('test visual features are nonfinite')
    missing_slots = sum(not np.any(valid & (slots == slot)) for slot in range(5))
    diagnostics = {'StudyInstanceUID': str(study_uid), 'valid_windows': int(valid.sum()),
                   'missing_slots': int(missing_slots), 'failed_series': len(failures),
                   'elapsed_seconds': time.perf_counter()-started,
                   'report_text_at_inference': False}
    return features, valid, slots, depths, diagnostics


def predict_study(models: list[MaskedSliceEncoder], features, valid, slots, depths,
                  *, device: torch.device) -> np.ndarray:
    if len(models) != 3:
        raise ValueError('exactly three fold models are required')
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0).to(device)
    mask = torch.from_numpy(np.asarray(valid, dtype=bool)).unsqueeze(0).to(device)
    slot = torch.from_numpy(np.asarray(slots, dtype=np.int64)).unsqueeze(0).to(device)
    depth = torch.from_numpy(np.asarray(depths, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores = [model(x, mask, slot, depth).sigmoid().squeeze(0).float().cpu().numpy()
                  for model in models]
    probability = np.mean(scores, axis=0)
    if probability.shape != (len(LABELS),) or not np.isfinite(probability).all():
        raise RuntimeError('test prediction is invalid')
    return probability
