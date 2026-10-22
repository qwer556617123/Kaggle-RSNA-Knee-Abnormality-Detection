from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from .artifacts import rank_columns


@dataclass(frozen=True)
class BlendResult:
    weights: np.ndarray  # [labels, models]
    prediction: np.ndarray
    per_label_auc: np.ndarray
    macro_auc: float
    best_single_macro_auc: float
    accepted: bool


def _simplex_candidates(models: int, step: float = 0.05) -> list[np.ndarray]:
    if models == 1:
        return [np.ones(1, dtype=np.float64)]
    units = int(round(1 / step))
    output = []

    def visit(prefix: list[int], left: int, count: int) -> None:
        if count == 1:
            output.append(np.asarray([*prefix, left], dtype=np.float64) / units)
            return
        for value in range(left + 1):
            visit([*prefix, value], left - value, count - 1)

    visit([], units, models)
    return output


def fit_rank_blend(
    targets: np.ndarray,
    predictions: Sequence[np.ndarray],
    *,
    minimum_gain: float = 0.003,
    shrink_to_equal: float = 0.20,
    grid_step: float = 0.05,
) -> BlendResult:
    targets = np.asarray(targets)
    ranked = np.stack([rank_columns(prediction) for prediction in predictions], axis=0)
    if ranked.ndim != 3 or ranked.shape[1:] != targets.shape:
        raise ValueError("all prediction matrices must match targets [rows, labels]")
    models, _, labels = ranked.shape
    candidates = _simplex_candidates(models, grid_step)
    equal = np.full(models, 1 / models, dtype=np.float64)
    weights = np.zeros((labels, models), dtype=np.float64)
    aucs = np.full(labels, np.nan, dtype=np.float64)

    single_scores = []
    for model_index in range(models):
        values = []
        for label in range(labels):
            if len(np.unique(targets[:, label])) == 2:
                values.append(roc_auc_score(targets[:, label], ranked[model_index, :, label]))
        single_scores.append(float(np.mean(values)) if values else np.nan)
    best_single = float(np.nanmax(single_scores))

    for label in range(labels):
        y = targets[:, label]
        if len(np.unique(y)) < 2:
            weights[label] = equal
            continue
        best_auc = -np.inf
        best_weight = equal
        for candidate in candidates:
            score = np.einsum("m,mn->n", candidate, ranked[:, :, label])
            auc = roc_auc_score(y, score)
            if auc > best_auc + 1e-12:
                best_auc, best_weight = auc, candidate
        stable_weight = (1 - shrink_to_equal) * best_weight + shrink_to_equal * equal
        weights[label] = stable_weight / stable_weight.sum()
        blended = np.einsum("m,mn->n", weights[label], ranked[:, :, label])
        aucs[label] = roc_auc_score(y, blended)

    prediction = np.column_stack(
        [np.einsum("m,mn->n", weights[label], ranked[:, :, label]) for label in range(labels)]
    ).astype(np.float32)
    macro = float(np.nanmean(aucs))
    return BlendResult(
        weights=weights,
        prediction=prediction,
        per_label_auc=aucs,
        macro_auc=macro,
        best_single_macro_auc=best_single,
        accepted=bool(macro >= best_single + minimum_gain),
    )


def apply_rank_blend(predictions: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    ranked = np.stack([rank_columns(prediction) for prediction in predictions], axis=0)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (ranked.shape[2], ranked.shape[0]):
        raise ValueError("weights must be [labels, models]")
    return np.column_stack(
        [np.einsum("m,mn->n", weights[label], ranked[:, :, label]) for label in range(ranked.shape[2])]
    ).astype(np.float32)
