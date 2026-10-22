import unittest

import torch
from torch import nn

from rsna_knee.anatomic_localizer import LearnedRegionMIL


class TinyEncoder(nn.Module):
    def __init__(self, dimension=8):
        super().__init__(); self.layer = nn.Linear(3, dimension)
    def forward(self, x):
        return self.layer(x.mean((-1, -2)))


class LocalizerTests(unittest.TestCase):
    def test_reader_reencodes_regions_and_exposes_ablation(self):
        model = LearnedRegionMIL(TinyEncoder(), TinyEncoder(), 8, local_size=8, global_size=8)
        images = torch.rand(2, 4, 3, 12, 12)
        logits, report = model(images, window_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool), return_diagnostics=True)
        self.assertEqual(tuple(logits.shape), (2, 12))
        self.assertEqual(tuple(report['region_centers'].shape), (2, 12, 2, 2))
        self.assertEqual(tuple(report['local_ablation_logits'].shape), (2, 12))
        self.assertGreaterEqual(float(report['continuity_loss']), 0.0)

    def test_missing_window_mask_is_rejected(self):
        model = LearnedRegionMIL(TinyEncoder(), TinyEncoder(), 8, local_size=8, global_size=8)
        with self.assertRaisesRegex(ValueError, 'matching'):
            model(torch.rand(1, 2, 3, 8, 8), window_mask=torch.ones(1, 3, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
