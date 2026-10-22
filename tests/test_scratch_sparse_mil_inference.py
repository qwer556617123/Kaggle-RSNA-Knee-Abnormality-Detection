import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from rsna_knee.artifacts import sha256_file, validate_submission
from rsna_knee.constants import LABELS
from rsna_knee.scratch_sparse_mil import ScratchSparseMIL
from rsna_knee.scratch_sparse_mil_inference import (
    RESEARCH_IDS, ROOT_ID, load_models, predict_study, prepare_series_table, prepare_study)


def _cache(*, missing=True, failed=False):
    pixels = np.zeros((24, 3, 336, 336), dtype=np.uint8)
    mask = np.ones(24, np.float32)
    if missing:
        mask[16:20] = 0
    metadata = [{"missing": not bool(value),
                 "source_detail": {"laterality": "L"} if value else {}}
                for value in mask]
    failures = [{"error": "decode failed"}] if failed else []
    return pixels, mask, metadata, failures


def test_test_inputs_are_report_free_and_missing_is_not_decode_failure():
    with patch("rsna_knee.scratch_sparse_mil_inference.build_study_cache", return_value=_cache()):
        inputs, detail = prepare_study("uid", pd.DataFrame(), "unused")
    assert inputs[0].shape == (1, 24, 3, 336, 336)
    assert inputs[4].item() == 0
    assert detail["missing_slots"] == 1
    assert detail["decode_failed_series"] == 0
    assert detail["report_text_at_inference"] is False
    with patch("rsna_knee.scratch_sparse_mil_inference.build_study_cache",
               return_value=_cache(failed=True)):
        with pytest.raises(RuntimeError, match="no fallback"):
            prepare_study("uid", pd.DataFrame(), "unused")


def test_series_requires_protocol_columns_and_identities():
    with pytest.raises(ValueError, match="lacks"):
        prepare_series_table(pd.DataFrame({"StudyInstanceUID": ["x"]}))
    frame = pd.DataFrame({"StudyInstanceUID": [None], "SeriesInstanceUID": ["s"],
                          "Anatomical_Plane": ["Sagittal"], "Fluid_Sensitive": [1]})
    with pytest.raises(ValueError, match="identities"):
        prepare_series_table(frame)


def _make_bundle(root):
    receipt_hashes, checkpoint_hashes = {}, {}
    for fold in range(3):
        folder = root / f"fold_{fold}"
        (folder / "candidate").mkdir(parents=True)
        checkpoint = folder / "candidate" / f"fold_{fold}.pt"
        torch.save({"model": ScratchSparseMIL(sparse_pool=True).state_dict(),
                    "fold": fold, "selected_epoch": 1}, checkpoint)
        checkpoint_hashes[str(fold)] = sha256_file(checkpoint)
        receipt = {"research_id": RESEARCH_IDS[fold],
                   "status": "pilot_completed" if fold == 0 else "paired_fold_completed",
                   "fold": fold, "source": "competition_MRI_pixels_only",
                   "external_pretrained_weights": False, "report_text_used": False,
                   "formal_submission_ready": False,
                   "cache_sha256": "c" * 64, "contract_version": "top10-clean-v1",
                   "evaluation_sha256": "e" * 64, "fold_sha256": "f" * 64,
                   "arms": {"candidate": {"status": "completed", "fold": fold,
                                           "selected_epoch": 1,
                                           "checkpoint_sha256": checkpoint_hashes[str(fold)]}}}
        if fold == 0:
            receipt["architecture"] = {"name": "scratch_target_aware_sparse_mil",
                                       "image_size": 160, "width": 128,
                                       "spatial_k": 5, "window_k": 2}
        else:
            receipt["parent_receipt_sha256"] = receipt_hashes["0"]
        path = folder / "run_manifest.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        receipt_hashes[str(fold)] = sha256_file(path)
    report = {"research_id": ROOT_ID, "source": "competition_MRI_pixels_only",
              "external_pretrained_weights": False, "mechanism_supported": True,
              "negative_screen_passed": True, "subgroup_review_complete": True,
              "robust_selection_eligible": True, "submission_ready": False,
              "parent_receipt_sha256": receipt_hashes["0"],
              "fold_receipt_sha256": receipt_hashes,
              "candidate_checkpoint_sha256_by_fold": checkpoint_hashes}
    (root / "effect_report.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def test_loading_requires_three_bound_checkpoints_and_robust_gate(tmp_path):
    report = _make_bundle(tmp_path)
    models, hashes, effect_hash = load_models(tmp_path, torch.device("cpu"))
    assert len(models) == 3 and hashes == report["candidate_checkpoint_sha256_by_fold"]
    assert effect_hash == sha256_file(tmp_path / "effect_report.json")
    checkpoint = tmp_path / "fold_2/candidate/fold_2.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="checkpoint SHA mismatch"):
        load_models(tmp_path, torch.device("cpu"))
    report["robust_selection_eligible"] = False
    (tmp_path / "effect_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="research gates"):
        load_models(tmp_path, torch.device("cpu"))


def test_threefold_prediction_and_csv_contract():
    models = [ScratchSparseMIL(sparse_pool=True) for _ in range(3)]
    inputs = (torch.zeros((1, 2, 3, 64, 64)), torch.ones((1, 2), dtype=torch.bool),
              torch.zeros((1, 2), dtype=torch.long), torch.zeros((1, 2)),
              torch.tensor([2]))
    probability = predict_study(models, inputs, device=torch.device("cpu"))
    assert probability.shape == (len(LABELS),)
    assert np.isfinite(probability).all()
    test = pd.DataFrame({"StudyInstanceUID": ["a", "b"]})
    csv = pd.DataFrame(np.stack([probability, probability]), columns=LABELS)
    csv.insert(0, "StudyInstanceUID", ["a", "b"])
    validate_submission(csv, test)
    with pytest.raises(ValueError, match="exactly three"):
        predict_study(models[:2], inputs, device=torch.device("cpu"))
