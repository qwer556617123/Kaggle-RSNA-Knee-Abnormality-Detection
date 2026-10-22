import pytest
import torch

from rsna_knee.report_pairwise import (
    assert_report_group_disjoint,
    group_balanced_pairwise_logistic_loss,
)


def _evidence(values, mask=None):
    values = torch.tensor(values, dtype=torch.float32)
    if values.ndim == 1:
        values = values[:, None]
    if mask is None:
        mask = torch.ones_like(values, dtype=torch.bool)
    return values, torch.as_tensor(mask, dtype=torch.bool)


def test_pairwise_loss_uses_same_image_logits_and_backpropagates():
    logits = torch.tensor([[3.0], [2.0], [-1.0], [-2.0]], requires_grad=True)
    values, mask = _evidence([1, 1, 0, 0])
    loss = group_balanced_pairwise_logistic_loss(logits, values, mask, ["p1", "p2", "n1", "n2"])

    assert loss.item() < 0.1
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] < 0  # increasing an explicit positive lowers loss
    assert logits.grad[2, 0] > 0  # increasing an explicit negative raises loss


def test_unknown_evidence_is_ignored_even_when_its_value_is_positive():
    logits = torch.tensor([[2.0], [-2.0], [-100.0]], requires_grad=True)
    values, mask = _evidence([1, 0, 1], [[True], [True], [False]])
    loss = group_balanced_pairwise_logistic_loss(logits, values, mask, ["p", "n", "unknown"])
    expected = torch.nn.functional.softplus(torch.tensor(-4.0))

    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad[2, 0].item() == 0.0


def test_group_balancing_prevents_duplicate_rows_from_changing_loss_weight():
    base_logits = torch.tensor([[1.0], [-1.0]])
    base_values, base_mask = _evidence([1, 0])
    base = group_balanced_pairwise_logistic_loss(base_logits, base_values, base_mask, ["positive", "negative"])

    duplicated_logits = torch.tensor([[1.0], [1.0], [-1.0]])
    duplicated_values, duplicated_mask = _evidence([1, 1, 0])
    duplicated = group_balanced_pairwise_logistic_loss(
        duplicated_logits, duplicated_values, duplicated_mask, ["positive", "positive", "negative"]
    )

    assert torch.allclose(base, duplicated)


def test_shuffling_batch_rows_with_their_evidence_and_groups_is_invariant():
    logits = torch.tensor([[2.0], [-2.0], [0.5], [-0.5]])
    values, mask = _evidence([1, 0, 1, 0])
    groups = ["p1", "n1", "p2", "n2"]
    original = group_balanced_pairwise_logistic_loss(logits, values, mask, groups)
    order = torch.tensor([2, 3, 0, 1])
    shuffled = group_balanced_pairwise_logistic_loss(
        logits[order], values[order], mask[order], [groups[index] for index in order.tolist()]
    )

    assert torch.allclose(original, shuffled)


def test_shuffled_evidence_negative_screen_is_worse_for_correct_logits():
    logits = torch.tensor([[3.0], [2.0], [-2.0], [-3.0]])
    values, mask = _evidence([1, 1, 0, 0])
    groups = ["g1", "g2", "g3", "g4"]
    aligned = group_balanced_pairwise_logistic_loss(logits, values, mask, groups)
    permuted_values = values[torch.tensor([2, 3, 0, 1])]
    shuffled = group_balanced_pairwise_logistic_loss(logits, permuted_values, mask, groups)

    assert shuffled > aligned


def test_same_report_group_pairs_and_partition_overlap_are_rejected_or_ignored():
    logits = torch.tensor([[2.0], [-2.0]], requires_grad=True)
    values, mask = _evidence([1, 0])
    loss = group_balanced_pairwise_logistic_loss(logits, values, mask, ["same", "same"])
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))

    with pytest.raises(ValueError, match="leakage"):
        assert_report_group_disjoint(["outer-train-a", "outer-train-b"], ["inner", "outer-train-b"])


def test_nonbinary_explicit_evidence_and_bad_group_lengths_fail_closed():
    logits = torch.zeros(2, 1)
    values, mask = _evidence([0.5, 0])
    with pytest.raises(ValueError, match="binary"):
        group_balanced_pairwise_logistic_loss(logits, values, mask, ["a", "b"])
    values, mask = _evidence([1, 0])
    with pytest.raises(ValueError, match="length"):
        group_balanced_pairwise_logistic_loss(logits, values, mask, ["a"])
