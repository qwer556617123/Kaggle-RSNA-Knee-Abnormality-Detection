"""Gold-independent robust consensus helpers for public weak-label sources."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConsensusResult:
    score: np.ndarray
    confidence: np.ndarray
    agreement: np.ndarray
    coverage: np.ndarray
    blend_alpha: np.ndarray


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = 0.5 * ordered_weights.sum()
    index = int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def robust_focal_consensus(
    incumbent_score: np.ndarray,
    incumbent_confidence: np.ndarray,
    source_scores: np.ndarray,
    source_confidence: np.ndarray | None = None,
    *,
    maximum_alpha: float = 0.65,
) -> ConsensusResult:
    """Blend an incumbent with source consensus without consulting gold labels."""
    incumbent_score = np.asarray(incumbent_score, dtype=np.float64)
    incumbent_confidence = np.asarray(incumbent_confidence, dtype=np.float64)
    source_scores = np.asarray(source_scores, dtype=np.float64)
    if source_scores.ndim != 2 or source_scores.shape[0] != len(incumbent_score):
        raise ValueError("source_scores must have shape [studies, sources]")
    if source_confidence is None:
        source_confidence = np.full_like(source_scores, 0.65)
    source_confidence = np.asarray(source_confidence, dtype=np.float64)
    if source_confidence.shape != source_scores.shape:
        raise ValueError("source_confidence shape mismatch")
    if not 0.0 <= maximum_alpha <= 1.0:
        raise ValueError("maximum_alpha must be in [0,1]")

    n_studies, n_sources = source_scores.shape
    score = incumbent_score.copy()
    confidence = incumbent_confidence.copy()
    agreement = np.zeros(n_studies, dtype=np.float64)
    coverage = np.zeros(n_studies, dtype=np.float64)
    blend_alpha = np.zeros(n_studies, dtype=np.float64)
    for index in range(n_studies):
        valid = np.isfinite(source_scores[index]) & np.isfinite(source_confidence[index])
        valid &= source_confidence[index] > 0
        if not valid.any():
            continue
        values = np.clip(source_scores[index, valid], 0.0, 1.0)
        weights = np.clip(source_confidence[index, valid], 0.05, 1.0)
        center = _weighted_median(values, weights)
        mean_abs_deviation = float(np.average(np.abs(values - center), weights=weights))
        agreement[index] = float(np.clip(1.0 - 2.0 * mean_abs_deviation, 0.0, 1.0))
        coverage[index] = float(valid.sum() / max(n_sources, 1))
        evidence_gate = np.clip((agreement[index] - 0.35) / 0.65, 0.0, 1.0)
        evidence_gate *= min(1.0, coverage[index] / 0.75)
        blend_alpha[index] = maximum_alpha * evidence_gate
        score[index] = (1.0 - blend_alpha[index]) * incumbent_score[index] + blend_alpha[index] * center
        certainty = float(np.clip(2.0 * abs(center - 0.5), 0.0, 1.0))
        consensus_confidence = np.clip(
            (0.15 + 0.45 * certainty + 0.40 * agreement[index]) * coverage[index],
            0.05,
            1.0,
        )
        confidence[index] = (
            (1.0 - blend_alpha[index]) * incumbent_confidence[index]
            + blend_alpha[index] * consensus_confidence
        )
    return ConsensusResult(
        score=np.clip(score, 0.0, 1.0).astype(np.float32),
        confidence=np.clip(confidence, 0.05, 1.0).astype(np.float32),
        agreement=agreement.astype(np.float32),
        coverage=coverage.astype(np.float32),
        blend_alpha=blend_alpha.astype(np.float32),
    )
