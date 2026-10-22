import pytest
import torch
from torch import nn
from rsna_knee.research_series import SeriesEvidenceMIL, from_control
from rsna_knee.research_coatnet import SpatialEvidenceMIL


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward_features(self, images):
        self.calls += 1
        return images


def test_conversion_preserves_control_initialization_and_rng():
    control = SpatialEvidenceMIL(Encoder(), 3, 'global')
    rng = torch.get_rng_state().clone()
    model = from_control(control)
    assert torch.equal(rng, torch.get_rng_state())
    assert model.backbone is control.backbone
    for name, value in control.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    assert set(model.state_dict()) - set(control.state_dict()) == {'slot_queries'}


def test_masks_gradients_and_encoder_budget():
    encoder = Encoder()
    model = SeriesEvidenceMIL(encoder, 3).eval()
    images = torch.randn(2, 6, 3, 4, 4)
    mask = torch.tensor([[1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0]], dtype=torch.bool)
    slots = torch.tensor([[0, 0, 1, 2, 3, 4]]).expand(2, -1)
    depth = torch.zeros(2, 6)
    side = torch.tensor([2, 0])
    result = model(images, mask, slots, depth, side)
    assert encoder.calls == 1
    assert torch.isfinite(result).all()
    changed = images.clone()
    changed[~mask] = 10000
    torch.testing.assert_close(result, model(changed, mask, slots, depth, side))
    order = torch.tensor([5, 2, 0, 4, 1, 3])
    torch.testing.assert_close(result, model(images[:, order], mask[:, order], slots[:, order], depth[:, order], side))
    result.sum().backward()
    assert model.slot_queries.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='no valid windows'):
        model(images, torch.zeros_like(mask), slots, depth, side)
