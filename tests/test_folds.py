import unittest

import numpy as np

from rsna_knee.folds import greedy_multilabel_group_folds, validate_group_folds
from rsna_knee.training import fold_calibrated_supervision


class FoldTests(unittest.TestCase):
    def test_persisted_group_folds_fail_when_a_group_crosses(self):
        with self.assertRaisesRegex(ValueError, "crosses folds"):
            validate_group_folds(np.array([0, 1, 2]), ["same", "same", "other"], n_splits=3)

    def test_persisted_group_folds_accept_complete_assignment(self):
        result = validate_group_folds(
            np.array([0.0, 0.0, 1.0, 2.0]), ["a", "a", "b", "c"], n_splits=3
        )
        np.testing.assert_array_equal(result, np.array([0, 0, 1, 2], dtype=np.int16))

    def test_groups_never_cross_folds(self):
        rng = np.random.default_rng(7)
        y = rng.uniform(size=(60, 12))
        groups = [f"g{index // 2}" for index in range(60)]
        folds = greedy_multilabel_group_folds(y, groups, n_splits=3)
        for group in set(groups):
            member_folds = {folds[index] for index, value in enumerate(groups) if value == group}
            self.assertEqual(len(member_folds), 1)
        self.assertEqual(set(folds), {0, 1, 2})
        self.assertLess(np.bincount(folds).max() - np.bincount(folds).min(), 12)

    def test_fold_reliability_cannot_read_validation_gold_or_raw_scores(self):
        targets = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
        confidence = np.full_like(targets, 0.8)
        gold = np.ones(4, dtype=bool)
        raw = np.array([[0.1], [0.9], [0.2], [0.8]], dtype=np.float32)
        _, first_confidence, first_reliability = fold_calibrated_supervision(
            targets, confidence, gold, raw, np.array([0, 1])
        )
        changed = raw.copy()
        changed[2:] = [[0.99], [0.01]]
        _, second_confidence, second_reliability = fold_calibrated_supervision(
            targets, confidence, gold, changed, np.array([0, 1])
        )
        self.assertTrue(np.array_equal(first_reliability, second_reliability))
        self.assertTrue(np.array_equal(first_confidence, second_confidence))

        # Even changing validation-side gold truth and membership cannot alter
        # reliability, because only train_indices [0, 1] are admissible.
        changed_targets = targets.copy()
        changed_targets[2:] = 1.0 - changed_targets[2:]
        changed_gold = gold.copy()
        changed_gold[2:] = False
        _, _, third_reliability = fold_calibrated_supervision(
            changed_targets, confidence, changed_gold, changed, np.array([0, 1])
        )
        self.assertTrue(np.array_equal(first_reliability, third_reliability))


if __name__ == "__main__":
    unittest.main()
