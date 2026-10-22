"""Regression tests for the new, isolated research validation contract."""
import io
import unittest

import numpy as np
import pandas as pd

from rsna_knee.research_evidence import evidence_weights
from rsna_knee.research_validation import (
    UID, acceptance_gate, masked_auc, nested_partition,
    paired_group_bootstrap, runtime_upper_bound, table_digest,
)


class ResearchValidationTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({UID: [f"u{i}" for i in range(180)],
                             "report_group": [f"g{i//2}" for i in range(180)],
                             "fold": [(i//2) % 3 for i in range(180)],
                             "is_gold": [1] + [0] * 179,
                             "truth": np.arange(180) % 2})

    def test_gold_group_and_outer_never_train(self):
        frame = self.frame()
        for fold in range(3):
            part = nested_partition(frame, fold)
            self.assertEqual(set(part.excluded), {0, 1})
            groups = [set(frame.iloc[p].report_group) for p in (part.train, part.inner, part.outer)]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            frame.loc[part.outer, "truth"] = 12345
            changed = nested_partition(frame, fold)
            np.testing.assert_array_equal(part.train, changed.train)
            np.testing.assert_array_equal(part.inner, changed.inner)

    def test_leaking_group_rejected(self):
        frame = self.frame()
        frame.loc[1, "fold"] = 1
        with self.assertRaises(ValueError):
            nested_partition(frame, 0)

    def test_invalid_indicators_rejected(self):
        for column, value in [("is_gold", "False"), ("fold", 3)]:
            frame = self.frame().astype({column: object})
            frame.loc[0, column] = value
            with self.assertRaises(ValueError):
                nested_partition(frame, 0)

    def test_csv_roundtrip_digest(self):
        frame = pd.DataFrame({"x": np.asarray([.123456789, .987654321], dtype=np.float32), "b": [True, False]})
        restored = pd.read_csv(io.StringIO(frame.to_csv(index=False, float_format="%.9g")))
        self.assertEqual(table_digest(frame), table_digest(restored))

    def test_paired_bootstrap_identical_is_zero(self):
        y = np.tile((np.arange(100) % 2)[:, None], (1, 12))
        pred = y * .6 + .2
        report = paired_group_bootstrap(y, pred, pred, np.ones_like(y), np.arange(100)//2, samples=20)
        self.assertEqual(report["macro_delta"], 0)
        self.assertEqual(report["ci95"], [0., 0.])

    def test_gate_requires_full_supported_comparison(self):
        report = {"macro_delta": .004, "ci95": [.001, .007]}
        self.assertFalse(acceptance_gate(report, [.004], complete_oof=False, compatible=True)["formal_submission_ready"])
        self.assertTrue(acceptance_gate(report, [.004, .006, -.001], complete_oof=True, compatible=True)["formal_submission_ready"])
        report["ci95"] = [-.001, .008]
        self.assertEqual(acceptance_gate(report, [.004]*3, complete_oof=True, compatible=True)["status"], "exploratory")

    def test_runtime_rejects_smoke_and_overbudget(self):
        with self.assertRaises(ValueError):
            runtime_upper_bound(20, [5]*3)
        self.assertTrue(runtime_upper_bound(100, [10]*96)["passes_7_5h"])
        self.assertFalse(runtime_upper_bound(100, [30]*96)["passes_7_5h"])

    def test_unresolved_not_explicit_negative(self):
        weights, states, _ = evidence_weights([1]*4, [[0,0,0], [0,0,0], [0,1,.5], [np.nan]*3], ["NO", "UNK", "YES", "UNK"])
        self.assertEqual(states[1], "unresolved_or_unmentioned")
        self.assertLess(weights[1], weights[0])
        self.assertLess(weights[2], weights[0])
        self.assertEqual(weights[3], 0)

    def test_invalid_evidence_and_mask_rejected(self):
        with self.assertRaises(ValueError):
            evidence_weights([np.nan], [[0,1]], ["NO"])
        with self.assertRaises(ValueError):
            masked_auc(np.zeros((2,12)), np.zeros((2,12)), np.full((2,12), np.nan))


if __name__ == "__main__":
    unittest.main()
