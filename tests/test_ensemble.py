import unittest

import numpy as np

from rsna_knee.ensemble import apply_rank_blend, fit_rank_blend


class EnsembleTests(unittest.TestCase):
    def test_nonnegative_simplex_weights(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=(100, 12))
        first = y + rng.normal(0, 0.7, size=y.shape)
        second = y + rng.normal(0, 0.9, size=y.shape)
        result = fit_rank_blend(y, [first, second], minimum_gain=0)
        self.assertEqual(result.weights.shape, (12, 2))
        self.assertTrue((result.weights >= 0).all())
        np.testing.assert_allclose(result.weights.sum(axis=1), 1)
        applied = apply_rank_blend([first, second], result.weights)
        np.testing.assert_allclose(applied, result.prediction)


if __name__ == "__main__":
    unittest.main()
