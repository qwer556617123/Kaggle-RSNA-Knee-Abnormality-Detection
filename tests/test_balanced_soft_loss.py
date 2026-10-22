import unittest
import numpy as np
import torch
from rsna_knee import balanced_soft_loss as loss
from rsna_knee.research_coatnet import soft_weighted_bce as control


class BalancedLossTests(unittest.TestCase):
    def test_balanced_data_exactly_recovers_control(self):
        loss.configure(np.full((4,12),.5),np.ones((4,12)))
        x=torch.linspace(-3,3,24).reshape(2,12).requires_grad_()
        y=torch.linspace(0,1,24).reshape(2,12);c=torch.full_like(y,.7)
        torch.testing.assert_close(loss.soft_weighted_bce(x,y,c),control(x,y,c))

    def test_training_mass_normalized_and_rare_positive_gradient_increased(self):
        y=np.zeros((100,12));y[:1]=1;c=np.ones_like(y)
        receipt=loss.configure(y,c)
        self.assertEqual(receipt['positive_weight'],[3.]*12)
        p=np.array(receipt['positive_mass']);n=np.array(receipt['negative_mass'])
        np.testing.assert_allclose((p*receipt['positive_weight']+n)/receipt['normalizer'],p+n)
        x=torch.zeros((1,12),requires_grad=True);target=torch.ones_like(x)
        grad=torch.autograd.grad(loss.soft_weighted_bce(x,target,target),x)[0]
        baseline=torch.autograd.grad(control(x,target,target),x)[0]
        self.assertTrue((grad.abs()>baseline.abs()).all())
        mask=torch.zeros_like(x)
        masked=torch.autograd.grad(loss.soft_weighted_bce(x,target,target,mask),x)[0]
        self.assertEqual(masked.abs().sum().item(),0)

    def test_invalid_class_support_rejected(self):
        with self.assertRaises(ValueError):loss.fit_weights(np.zeros((4,12)),np.ones((4,12)))
