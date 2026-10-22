import numpy as np


import pytest


import torch


from rsna_knee.scratch_sparse_mil import ScratchSparseMIL


from rsna_knee.constants import LABELS


def test_paired_parameters_and_missing_sequence_gradients():
    torch.manual_seed(13)
    control = ScratchSparseMIL(sparse_pool=False, width=32, image_size=64)
    torch.manual_seed(13)
    candidate = ScratchSparseMIL(sparse_pool=True, width=32, image_size=64)
    assert all(torch.equal(a, b) for a, b in zip(control.parameters(), candidate.parameters()))
    images = torch.randn(2, 5, 3, 64, 64)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    slots = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 1, 3, 4]])
    depth = torch.zeros(2, 5)
    laterality = torch.tensor([0, 2])
    for instance in (control, candidate):
        output = instance(images, window_mask=mask, slot_ids=slots,
                          depth=depth, laterality=laterality)
        assert output.shape == (2, 12) and torch.isfinite(output).all()
        output.square().mean().backward()
        gradient = instance.backbone[0].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_invalid_and_masked_windows_cannot_silently_affect_prediction():
    torch.manual_seed(3)
    model = ScratchSparseMIL(sparse_pool=True, width=32, image_size=64).eval()
    images = torch.randn(1, 3, 3, 64, 64)
    mask = torch.tensor([[1, 0, 0]], dtype=torch.bool)
    slots = torch.tensor([[1, 2, 3]])
    depth = torch.zeros(1, 3)
    laterality = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        first = model(images, window_mask=mask, slot_ids=slots,
                      depth=depth, laterality=laterality)
        images[:, 1:] *= 100
        second = model(images, window_mask=mask, slot_ids=slots,
                       depth=depth, laterality=laterality)
    assert torch.allclose(first, second, atol=1e-6)
    with pytest.raises(ValueError, match="metadata"):
        model(images, window_mask=torch.zeros_like(mask), slot_ids=slots,
              depth=depth, laterality=laterality)
