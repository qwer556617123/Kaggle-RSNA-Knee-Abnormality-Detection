from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from rsna_knee.scratch_cross_sequence_inference import prepare_series_table, prepare_study, predict_study


def test_series_table_needs_protocol_fields():
    with pytest.raises(ValueError, match="lacks"):
        prepare_series_table(pd.DataFrame({"StudyInstanceUID": ["x"]}))


def test_report_free_inputs_distinguish_missing_slot_from_decode_failure():
    pixels = np.zeros((24, 3, 336, 336), dtype=np.uint8)
    mask = np.ones(24, np.float32)
    mask[16:20] = 0
    metadata = [{"missing": not bool(value),
                 "source_detail": {"laterality": "L"} if value else {}}
                for value in mask]
    with patch("rsna_knee.scratch_cross_sequence_inference.build_study_cache",
               return_value=(pixels, mask, metadata, [])):
        inputs, diagnostics = prepare_study("uid", pd.DataFrame(), "unused")
    assert diagnostics["missing_slots"] == 1
    assert diagnostics["decode_failed_series"] == 0
    assert diagnostics["laterality"] == "L"
    assert inputs[0].shape == (1, 24, 3, 336, 336)
    assert inputs[4].item() == 0
    with patch("rsna_knee.scratch_cross_sequence_inference.build_study_cache",
               return_value=(pixels, mask, metadata, [{"error": "decode failed"}])):
        with pytest.raises(RuntimeError, match="no fallback"):
            prepare_study("uid", pd.DataFrame(), "unused")


def test_predict_requires_three_candidate_models():
    with pytest.raises(ValueError, match="exactly three"):
        predict_study([], (), device=torch.device("cpu"))
