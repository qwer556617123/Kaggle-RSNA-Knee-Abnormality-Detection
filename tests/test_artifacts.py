import unittest

import numpy as np
import pandas as pd

from rsna_knee.artifacts import validate_feature_artifact, validate_submission, write_json
from rsna_knee.constants import LABELS


class SubmissionTests(unittest.TestCase):
    def test_valid_submission(self):
        test = pd.DataFrame({"StudyInstanceUID": ["a", "b", "c", "d"]})
        submission = test.copy()
        for label in LABELS:
            submission[label] = np.linspace(0.1, 0.9, len(test))
        self.assertIs(validate_submission(submission, test), submission)

    def test_rejects_wrong_order_and_nan(self):
        test = pd.DataFrame({"StudyInstanceUID": ["a", "b"]})
        submission = test.iloc[::-1].reset_index(drop=True)
        for label in LABELS:
            submission[label] = 0.5
        with self.assertRaises(ValueError):
            validate_submission(submission, test)
        submission = test.copy()
        for label in LABELS:
            submission[label] = 0.5
        submission.loc[0, LABELS[0]] = np.nan
        with self.assertRaises(ValueError):
            validate_submission(submission, test)

    def test_write_json_serializes_numpy_bool(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_json(path, {"complete_oof": np.bool_(True)})
            self.assertIn("true", path.read_text(encoding="utf-8"))

    def test_feature_artifact_contract(self):
        ids = ["a", "b", "c"]
        features = np.ones((3, 5, 9, 16), dtype=np.float16)
        mask = np.ones((3, 5), dtype=np.uint8)
        self.assertEqual(validate_feature_artifact(features, mask, ids, ids), 16)
        features[1, 2, 3, 4] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN/inf"):
            validate_feature_artifact(features, mask, ids, ids)

    def test_feature_artifact_rejects_alignment_and_empty_study(self):
        features = np.ones((2, 5, 9, 8), dtype=np.float32)
        mask = np.ones((2, 5), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "align exactly"):
            validate_feature_artifact(features, mask, ["b", "a"], ["a", "b"])
        mask[1] = 0
        with self.assertRaisesRegex(ValueError, "no decoded protocol"):
            validate_feature_artifact(features, mask, ["a", "b"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
