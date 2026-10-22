from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def greedy_multilabel_group_folds(
    y: np.ndarray,
    groups: Iterable[str],
    n_splits: int = 3,
    seed: int = 20260807,
) -> np.ndarray:
    """Assign whole groups to folds while balancing soft multilabel mass and size.

    The result is deterministic and never splits duplicate report fingerprints.
    Soft targets are supported because report-derived labels are intentionally graded.
    """
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(list(groups), dtype=object)
    if y.ndim != 2 or len(y) != len(groups):
        raise ValueError("y must be [n_samples, n_labels] and align with groups")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    members: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        members[str(group)].append(index)

    rng = np.random.default_rng(seed)
    items = []
    for group, indices in members.items():
        mass = y[indices].sum(axis=0)
        rarity = float(np.sum(mass / np.maximum(y.sum(axis=0), 1e-6)))
        items.append((group, indices, mass, rarity, rng.random()))
    items.sort(key=lambda item: (-item[3], -len(item[1]), item[4]))

    fold_mass = np.zeros((n_splits, y.shape[1]), dtype=np.float64)
    fold_size = np.zeros(n_splits, dtype=np.float64)
    target_mass = y.sum(axis=0) / n_splits
    target_size = len(y) / n_splits
    assignment = np.full(len(y), -1, dtype=np.int16)

    for _, indices, mass, _, _ in items:
        costs = []
        for fold in range(n_splits):
            next_mass = fold_mass.copy()
            next_size = fold_size.copy()
            next_mass[fold] += mass
            next_size[fold] += len(indices)
            label_cost = np.mean(
                ((next_mass - target_mass) / np.maximum(target_mass, 1.0)) ** 2
            )
            size_cost = np.mean(((next_size - target_size) / max(target_size, 1.0)) ** 2)
            costs.append(label_cost + 0.20 * size_cost)
        chosen = int(np.argmin(costs))
        assignment[indices] = chosen
        fold_mass[chosen] += mass
        fold_size[chosen] += len(indices)

    if (assignment < 0).any():
        raise RuntimeError("fold assignment is incomplete")
    return assignment

def validate_group_folds(
    assignment: np.ndarray,
    groups: Iterable[str],
    *,
    n_splits: int,
) -> np.ndarray:
    """Fail closed when a persisted assignment is incomplete or splits a report group."""
    assignment = np.asarray(assignment)
    groups = np.asarray(list(groups), dtype=object)
    if assignment.ndim != 1 or len(assignment) != len(groups):
        raise ValueError("fold assignment must be one-dimensional and align with groups")
    try:
        numeric = assignment.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("fold assignment must contain numeric fold IDs") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("fold assignment must contain integer fold IDs")
    result = numeric.astype(np.int16)
    if (result < 0).any() or (result >= n_splits).any():
        raise ValueError("fold assignment contains an out-of-range fold ID")
    if set(result.tolist()) != set(range(n_splits)):
        raise ValueError("fold assignment must populate every fold")
    seen: dict[str, int] = {}
    for group, fold in zip(groups, result):
        key = str(group)
        previous = seen.setdefault(key, int(fold))
        if previous != int(fold):
            raise ValueError(f"report group crosses folds: {key}")
    return result
