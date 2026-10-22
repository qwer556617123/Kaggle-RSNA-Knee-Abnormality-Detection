"""Report-free test inputs and hash-bound three-fold scratch MIL loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifacts import sha256_file
from .constants import LABELS, SLOT_NAMES
from .dicom import slot_for_series
from .research_cache import ResearchCacheConfig, build_study_cache
from .scratch_sparse_mil import ScratchSparseMIL

COUNTS = (4, 6, 6, 4, 4)
RESEARCH_IDS = ("scratch-sparse-mil-v1", "scratch-sparse-mil-fold1-v1",
                "scratch-sparse-mil-fold2-v1")
ROOT_ID = "scratch-sparse-mil-threefold-v1"


def bundle_sha256(root: str | Path) -> str:
    """Hash seven exact input files in fixed path order."""
    root = Path(root)
    names = ["effect_report.json"]
    for fold in range(3):
        names.extend((f"fold_{fold}/run_manifest.json",
                      f"fold_{fold}/candidate/fold_{fold}.pt"))
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("ascii") + b"\0" + sha256_file(root / name).encode("ascii") + b"\n")
    return digest.hexdigest()


def prepare_series_table(series: pd.DataFrame) -> pd.DataFrame:
    required = {"StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"}
    if not required <= set(series):
        raise ValueError(f"test series table lacks {sorted(required - set(series))}")
    table = series.copy()
    if table[["StudyInstanceUID", "SeriesInstanceUID"]].isna().any().any():
        raise ValueError("test series contains missing identities")
    table["StudyInstanceUID"] = table.StudyInstanceUID.astype(str)
    table["SeriesInstanceUID"] = table.SeriesInstanceUID.astype(str)
    table["slot"] = [slot_for_series(plane, fluid)
                     for plane, fluid in zip(table.Anatomical_Plane, table.Fluid_Sensitive)]
    return table.loc[table.slot.isin(SLOT_NAMES)].copy()


def prepare_study(study_uid: str, series: pd.DataFrame, dicom_root: str | Path):
    """Use valid MRI windows as in the frozen training cache; track rejected series."""
    pixels, mask, metadata, failures = build_study_cache(
        str(study_uid), series, dicom_root, ResearchCacheConfig())
    if (pixels.shape != (24, 3, 336, 336) or pixels.dtype != np.uint8
            or mask.shape != (24,) or len(metadata) != 24):
        raise RuntimeError("test DICOM deviates from frozen 24-window cache geometry")
    if not np.isin(mask, (0, 1)).all() or not mask.any():
        raise RuntimeError("study has no valid competition MRI windows")
    slots = np.repeat(np.arange(5, dtype=np.int64), COUNTS)
    valid = mask.astype(bool)
    if any(bool(item.get("missing")) == bool(valid[index]) for index, item in enumerate(metadata)):
        raise RuntimeError("DICOM cache mask and metadata disagree")
    depths = np.concatenate([np.linspace(0, 1, count, dtype=np.float32) for count in COUNTS])
    sides = {str(item.get("source_detail", {}).get("laterality", "unknown")).upper()
             for index, item in enumerate(metadata) if valid[index]}
    side = 0 if sides == {"L"} else 1 if sides == {"R"} else 2
    image = torch.from_numpy(pixels.astype(np.float32) / 255.).sub(.5).div(.5).unsqueeze(0)
    inputs = (image, torch.from_numpy(valid).unsqueeze(0),
              torch.from_numpy(slots).unsqueeze(0),
              torch.from_numpy(depths).unsqueeze(0),
              torch.tensor([side], dtype=torch.long))
    detail = {"StudyInstanceUID": str(study_uid), "valid_windows": int(valid.sum()),
              "missing_slots": int(sum(not np.any(valid & (slots == slot)) for slot in range(5))),
              "decode_failed_series": len(failures), "study_fallback": False,
              "failed_candidate_slots": ",".join(sorted({str(item["slot"]) for item in failures})),
              "laterality": next(iter(sides)) if len(sides) == 1 else "unknown",
              "report_text_at_inference": False}
    return inputs, detail


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON receipt: {path}")
    return value


def load_models(root: str | Path, device: torch.device):
    """Accept only a complete, independently reviewed three-fold research bundle.

    Bundle: effect_report.json and fold_{0,1,2}/{run_manifest.json,
    candidate/fold_N.pt}. The report and each receipt bind exact checkpoint bytes.
    """
    root = Path(root)
    report_path = root / "effect_report.json"
    report = _read_json(report_path)
    if (report.get("research_id") != ROOT_ID
            or report.get("source") != "competition_MRI_pixels_only"
            or report.get("external_pretrained_weights") is not False
            or report.get("mechanism_supported") is not True
            or report.get("negative_screen_passed") is not True
            or report.get("subgroup_review_complete") is not True
            or report.get("robust_selection_eligible") is not True
            or report.get("submission_ready") is not False):
        raise RuntimeError("three-fold research gates or rights review have not passed")
    receipts, models, hashes = [], [], {}
    parent_sha = report.get("parent_receipt_sha256")
    for fold in range(3):
        folder = root / f"fold_{fold}"
        receipt_path = folder / "run_manifest.json"
        receipt_sha = sha256_file(receipt_path)
        if receipt_sha != report.get("fold_receipt_sha256", {}).get(str(fold)):
            raise RuntimeError(f"fold {fold} research receipt SHA mismatch")
        receipt = _read_json(receipt_path)
        if (receipt.get("research_id") != RESEARCH_IDS[fold]
                or receipt.get("status") != ("pilot_completed" if fold == 0 else "paired_fold_completed")
                or receipt.get("fold") != fold
                or receipt.get("source") != "competition_MRI_pixels_only"
                or receipt.get("external_pretrained_weights") is not False
                or receipt.get("report_text_used") is not False
                or receipt.get("formal_submission_ready") is not False
                or (fold > 0 and receipt.get("parent_receipt_sha256") != parent_sha)):
            raise RuntimeError(f"fold {fold} lineage differs from competition-only research")
        if fold == 0 and receipt_sha != parent_sha:
            raise RuntimeError("fold-0 parent receipt SHA mismatch")
        candidate = receipt.get("arms", {}).get("candidate", {})
        if candidate.get("status") != "completed" or candidate.get("fold") != fold:
            raise RuntimeError(f"fold {fold} candidate training incomplete")
        checkpoint = folder / "candidate" / f"fold_{fold}.pt"
        digest = sha256_file(checkpoint)
        if (digest != candidate.get("checkpoint_sha256")
                or digest != report.get("candidate_checkpoint_sha256_by_fold", {}).get(str(fold))):
            raise RuntimeError(f"fold {fold} checkpoint SHA mismatch")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (not isinstance(state, dict) or state.get("fold") != fold
                or state.get("selected_epoch") != candidate.get("selected_epoch")):
            raise RuntimeError(f"fold {fold} checkpoint identity mismatch")
        model = ScratchSparseMIL(sparse_pool=True, targets=len(LABELS), width=128, image_size=160)
        model.load_state_dict(state["model"], strict=True)
        model.to(device).eval()
        receipts.append(receipt)
        models.append(model)
        hashes[str(fold)] = digest
    same = ("cache_sha256", "contract_version", "evaluation_sha256", "fold_sha256")
    if any(any(item.get(key) != receipts[0].get(key) for key in same) for item in receipts[1:]):
        raise RuntimeError("fold training contracts disagree")
    arch = receipts[0].get("architecture", {})
    if (arch.get("name") != "scratch_target_aware_sparse_mil" or arch.get("image_size") != 160
            or arch.get("width") != 128 or arch.get("spatial_k") != ScratchSparseMIL.SPATIAL_K
            or arch.get("window_k") != ScratchSparseMIL.WINDOW_K):
        raise RuntimeError("fold-0 architecture differs from inference source")
    return models, hashes, sha256_file(report_path)


def predict_study(models: list[ScratchSparseMIL], inputs, *, device: torch.device) -> np.ndarray:
    if len(models) != 3 or any(not isinstance(model, ScratchSparseMIL) or not model.sparse_pool
                               for model in models):
        raise ValueError("exactly three candidate scratch sparse MIL folds are required")
    image, mask, slots, depths, laterality = [item.to(device) for item in inputs]
    scores = []
    with torch.inference_mode():
        for model in models:
            model.eval()
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(image, window_mask=mask, slot_ids=slots,
                               depth=depths, laterality=laterality)
            scores.append(logits.float().sigmoid().squeeze(0).cpu().numpy())
    probability = np.mean(scores, axis=0)
    if probability.shape != (len(LABELS),) or not np.isfinite(probability).all():
        raise RuntimeError("three-fold test prediction is invalid")
    return probability
