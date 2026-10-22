import unittest

import numpy as np

from rsna_knee.label_consensus import robust_focal_consensus


class LabelConsensusTests(unittest.TestCase):
    def test_agreeing_sources_move_incumbent_toward_consensus(self):
        result = robust_focal_consensus(
            np.array([0.20]), np.array([0.70]),
            np.array([[0.80, 0.82, 0.78, 0.81]]), np.ones((1, 4)),
        )
        self.assertGreater(result.score[0], 0.20)
        self.assertGreater(result.blend_alpha[0], 0.60)
        self.assertGreater(result.agreement[0], 0.90)

    def test_disagreement_suppresses_update(self):
        result = robust_focal_consensus(
            np.array([0.40]), np.array([0.80]),
            np.array([[0.0, 1.0, 0.0, 1.0]]), np.ones((1, 4)),
        )
        self.assertAlmostEqual(float(result.score[0]), 0.40, places=6)
        self.assertAlmostEqual(float(result.blend_alpha[0]), 0.0, places=6)

    def test_missing_sources_do_not_manufacture_negative(self):
        result = robust_focal_consensus(
            np.array([0.73]), np.array([0.45]),
            np.full((1, 4), np.nan), np.full((1, 4), np.nan),
        )
        self.assertAlmostEqual(float(result.score[0]), 0.73, places=6)
        self.assertAlmostEqual(float(result.confidence[0]), 0.45, places=6)

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            robust_focal_consensus(
                np.array([0.5]), np.array([0.5]), np.ones((1, 2)), np.ones((1, 3))
            )


if __name__ == "__main__":
    unittest.main()
