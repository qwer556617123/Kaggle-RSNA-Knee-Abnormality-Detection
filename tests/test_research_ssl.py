import unittest
import torch
from rsna_knee.research_ssl import MRIBarlowEncoder, augment_mri_windows, barlow_loss


class ResearchSSLTests(unittest.TestCase):
    def test_barlow_gradients_reach_scratch_backbone(self):
        torch.manual_seed(7)
        model = MRIBarlowEncoder(image_size=64, width=32, projection_dim=16)
        images = torch.rand(4, 3, 64, 64).mul(2).sub(1)
        first = augment_mri_windows(images)
        second = augment_mri_windows(images)
        loss = barlow_loss(model(first), model(second))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        grad = model.backbone[0].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0)
