"""Fold-local, report-derived pairwise ranking supervision.

This module deliberately accepts only image-model logits plus already-extracted
explicit evidence.  It never receives report text and is therefore safe to use
from image-only validation and inference code.  Callers are responsible for
extracting evidence from the outer-training partition only.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch.nn import functional as F


def assert_report_group_disjoint(
    training_groups: Iterable[object], evaluation_groups: Iterable[object],
) -> None:
    """Reject report-group leakage between a training and held-out partition."""
    train = {str(value) for value in training_groups}
    evaluation = {str(value) for value in evaluation_groups}
    overlap = train & evaluation
    if overlap:
        preview = ", ".join(sorted(overlap)[:3])
        raise ValueError(f"report-group leakage between training and held-out rows: {preview}")


def _normalise_groups(report_groups: Sequence[object], batch_size: int) -> list[str]:
    if isinstance(report_groups, (str, bytes)):
        raise ValueError("report_groups must contain one value per batch row")
    groups = [str(value) for value in report_groups]
    if len(groups) != batch_size:
        raise ValueError("report_groups length must match logits batch size")
    if any(not value.strip() for value in groups):
        raise ValueError("report_groups cannot contain empty values")
    return groups


def _validate_inputs(
    logits: torch.Tensor,
    evidence_values: torch.Tensor,
    evidence_mask: torch.Tensor,
    report_groups: Sequence[object],
) -> list[str]:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, targets]")
    if evidence_values.shape != logits.shape or evidence_mask.shape != logits.shape:
        raise ValueError("explicit evidence values and mask must match logits")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must be floating point")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    mask = evidence_mask.bool()
    if mask.any():
        values = evidence_values[mask]
        if not torch.isfinite(values).all() or not torch.all((values == 0) | (values == 1)):
            raise ValueError("explicit evidence values must be binary where evidence_mask is true")
    return _normalise_groups(report_groups, logits.shape[0])


def group_balanced_pairwise_logistic_loss(
    logits: torch.Tensor,
    evidence_values: torch.Tensor,
    evidence_mask: torch.Tensor,
    report_groups: Sequence[object],
    *,
    margin: float = 0.0,
) -> torch.Tensor:
    """Return a group-balanced positive-vs-negative logistic ranking loss.

    ``evidence_mask`` defines which report-derived states are explicit. Unknown,
    uncertain, and conflicting report states must have a false mask and are
    ignored regardless of the accompanying value.  For each target, every
    positive report group is compared with every *different* negative group.
    Each group-pair contributes one mean loss, so studies duplicated within a
    report group cannot dominate the objective.  The target losses are then
    averaged equally.  No per-example report representation enters this API.

    If a batch has no eligible cross-group positive/negative pairs, a
    differentiable zero is returned.  This lets a regular training batch remain
    valid while contributing no pairwise signal.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    groups = _validate_inputs(logits, evidence_values, evidence_mask, report_groups)
    mask = evidence_mask.bool()
    values = evidence_values.to(device=logits.device)
    target_losses: list[torch.Tensor] = []

    for target in range(logits.shape[1]):
        explicit = mask[:, target]
        positive_rows = torch.where(explicit & (values[:, target] == 1))[0]
        negative_rows = torch.where(explicit & (values[:, target] == 0))[0]
        if positive_rows.numel() == 0 or negative_rows.numel() == 0:
            continue

        positive_by_group: dict[str, list[int]] = {}
        negative_by_group: dict[str, list[int]] = {}
        for row in positive_rows.detach().cpu().tolist():
            positive_by_group.setdefault(groups[row], []).append(row)
        for row in negative_rows.detach().cpu().tolist():
            negative_by_group.setdefault(groups[row], []).append(row)

        group_pair_losses: list[torch.Tensor] = []
        for positive_group, positive_indices in positive_by_group.items():
            positive_scores = logits[positive_indices, target]
            for negative_group, negative_indices in negative_by_group.items():
                if positive_group == negative_group:
                    # A report group is a leakage boundary, not a source of an
                    # apparently independent positive/negative comparison.
                    continue
                negative_scores = logits[negative_indices, target]
                differences = positive_scores[:, None] - negative_scores[None, :]
                group_pair_losses.append(F.softplus(margin - differences).mean())
        if group_pair_losses:
            target_losses.append(torch.stack(group_pair_losses).mean())

    return torch.stack(target_losses).mean() if target_losses else logits.sum() * 0.0
