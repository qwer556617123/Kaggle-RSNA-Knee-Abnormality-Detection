import torch

from rsna_knee.scratch_cross_sequence import ScratchCrossSequence


def test_attention_edges_are_only_same_plane_pairs():
    slots = torch.tensor([[0, 1, 2, 3, 4]])
    control = ScratchCrossSequence(cross_pair=False, width=32, image_size=64)
    candidate = ScratchCrossSequence(cross_pair=True, width=32, image_size=64)
    same = control.attention_allowed(slots)[0]
    cross = candidate.attention_allowed(slots)[0]
    assert same.diagonal().all() and same.sum() == 5
    assert cross[1, 3] and cross[3, 1] and cross[2, 4] and cross[4, 2]
    assert not cross[0, 1] and not cross[1, 4]


def test_identical_parameters_and_missing_slot_finite_backward():
    torch.manual_seed(7)
    control = ScratchCrossSequence(cross_pair=False, width=32, image_size=64)
    torch.manual_seed(7)
    candidate = ScratchCrossSequence(cross_pair=True, width=32, image_size=64)
    assert all(torch.equal(a, b) for a, b in zip(control.parameters(), candidate.parameters()))
    images = torch.randn(1, 5, 3, 64, 64)
    mask = torch.tensor([[1, 1, 0, 1, 0]], dtype=torch.bool)
    slot = torch.tensor([[0, 1, 2, 3, 4]])
    depth = torch.zeros(1, 5)
    laterality = torch.ones(1, dtype=torch.long)
    output = candidate(images, window_mask=mask, slot_ids=slot, depth=depth, laterality=laterality)
    assert output.shape == (1, 12) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert candidate.backbone[0].weight.grad is not None
    assert torch.isfinite(candidate.backbone[0].weight.grad).all()
