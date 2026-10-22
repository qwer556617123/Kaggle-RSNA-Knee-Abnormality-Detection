import unittest

import torch

from rsna_knee.masked_slice import MaskedSliceEncoder


class MaskedSliceTests(unittest.TestCase):
    def test_contiguous_fold_local_mask_and_reconstruction_gradients(self):
        model = MaskedSliceEncoder(16, hidden_dim=32)
        features = torch.randn(2, 24, 16)
        valid = torch.ones(2, 24, dtype=torch.bool)
        slots = torch.tensor([0]*4 + [1]*6 + [2]*6 + [3]*4 + [4]*4).repeat(2, 1)
        depth = torch.linspace(0, 1, 24).repeat(2, 1)
        selected = model.block_mask(valid, slots, seed=7)
        self.assertTrue(torch.equal(selected, model.block_mask(valid, slots, seed=7)))
        self.assertGreaterEqual(float(selected.float().mean()), .20)
        self.assertLessEqual(float(selected.float().mean()), .30)
        loss, diagnostics = model.reconstruction_loss(features, valid, slots, depth, seed=7)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.project.weight.grad)
        self.assertEqual(diagnostics['masked_windows'], int(selected.sum()))
        self.assertEqual(tuple(model(features, valid, slots, depth).shape), (2, 12))

    def test_missing_study_is_rejected(self):
        model = MaskedSliceEncoder(8, hidden_dim=32)
        with self.assertRaisesRegex(ValueError, 'no valid'):
            model(torch.zeros(1, 4, 8), torch.zeros(1, 4, dtype=torch.bool),
                  torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4))
