"""Report-free MRI inputs for a future three-fold scratch deployment rehearsal.

This module does not load checkpoints or authorize a competition submission.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import LABELS, SLOT_NAMES
from .dicom import slot_for_series
from .research_cache import ResearchCacheConfig, build_study_cache
from .scratch_cross_sequence import ScratchCrossSequence

COUNTS = (4, 6, 6, 4, 4)


def prepare_series_table(series: pd.DataFrame) -> pd.DataFrame:
    required = {"StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"}
    if not required <= set(series):
        raise ValueError(f"test series table lacks {sorted(required-set(series))}")
    table = series.copy()
    table["StudyInstanceUID"] = table.StudyInstanceUID.astype(str)
    table["SeriesInstanceUID"] = table.SeriesInstanceUID.astype(str)
    table["slot"] = [slot_for_series(plane, fluid)
                     for plane, fluid in zip(table.Anatomical_Plane, table.Fluid_Sensitive)]
    return table.loc[table.slot.isin(SLOT_NAMES)].copy()


def prepare_study(study_uid: str, series: pd.DataFrame, dicom_root: str | Path):
    pixels, mask, metadata, failures = build_study_cache(
        str(study_uid), series, dicom_root, ResearchCacheConfig())
    if pixels.shape != (24, 3, 336, 336) or pixels.dtype != np.uint8 or mask.shape != (24,):
        raise RuntimeError("test DICOM deviates from frozen 24-window cache geometry")
    if not np.isin(mask, [0, 1]).all() or not mask.any():
        raise RuntimeError("study has no valid competition MRI windows")
    if failures:
        raise RuntimeError(f"test DICOM series decode/geometry failure: {len(failures)}; no fallback")
    valid = mask.astype(bool)
    slots = np.repeat(np.arange(5, dtype=np.int64), COUNTS)
    depths = np.concatenate([np.linspace(0, 1, n, dtype=np.float32) for n in COUNTS])
    laterality_values = {str(item.get("source_detail", {}).get("laterality", "unknown")).upper()
                         for item in metadata if not item.get("missing")}
    side = 0 if laterality_values == {"L"} else 1 if laterality_values == {"R"} else 2
    image = torch.from_numpy(pixels.astype(np.float32) / 255.).sub(.5).div(.5).unsqueeze(0)
    inputs = (image, torch.from_numpy(valid).unsqueeze(0),
              torch.from_numpy(slots).unsqueeze(0),
              torch.from_numpy(depths).unsqueeze(0),
              torch.tensor([side], dtype=torch.long))
    diagnostics = {"StudyInstanceUID": str(study_uid), "valid_windows": int(valid.sum()),
                   "missing_slots": int(sum(not np.any(valid & (slots == slot)) for slot in range(5))),
                   "decode_failed_series": 0, "study_fallback": False,
                   "laterality": laterality_values.pop() if len(laterality_values) == 1 else "unknown",
                   "report_text_at_inference": False}
    return inputs, diagnostics


def predict_study(models: list[ScratchCrossSequence], inputs, *, device: torch.device) -> np.ndarray:
    if len(models) != 3 or any(not isinstance(item, ScratchCrossSequence) or not item.cross_pair
                               for item in models):
        raise ValueError("exactly three candidate scratch cross-sequence folds are required")
    image, mask, slots, depths, laterality = [item.to(device) for item in inputs]
    scores = []
    with torch.inference_mode():
        for model in models:
            model.eval()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(image, window_mask=mask, slot_ids=slots,
                               depth=depths, laterality=laterality)
            scores.append(logits.float().sigmoid().squeeze(0).cpu().numpy())
    probability = np.mean(scores, axis=0)
    if probability.shape != (len(LABELS),) or not np.isfinite(probability).all():
        raise RuntimeError("three-fold test prediction is invalid")
    return probability
