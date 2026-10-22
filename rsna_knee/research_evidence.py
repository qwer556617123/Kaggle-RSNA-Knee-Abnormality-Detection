"""Evidence-aware confidence ablation; probability targets remain unchanged."""
from __future__ import annotations

import numpy as np


def evidence_weights(incumbent_confidence, family_scores, verdict, *, conflict_scale=0.75):
    scores = np.asarray(family_scores, dtype=float)
    base = np.asarray(incumbent_confidence, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != len(base) or len(verdict) != len(base):
        raise ValueError("evidence inputs must align")
    if base.ndim != 1 or not np.isfinite(base).all() or ((base < 0) | (base > 1)).any() or not 0 <= conflict_scale <= 1:
        raise ValueError("invalid confidence or conflict scale")
    if np.isinf(scores).any() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("invalid source probabilities")
    valid = np.isfinite(scores)
    counts = valid.sum(axis=1)
    # A source family gets one vote, regardless of how many versions it has.
    safe = np.where(valid, scores, 0)
    mean = safe.sum(axis=1) / np.maximum(counts, 1)
    disagreement = np.sqrt((np.where(valid, (safe-mean[:, None])**2, 0)).sum(axis=1) / np.maximum(counts, 1))
    agreement = np.clip(1 - 2*disagreement, 0, 1)
    verdict = np.asarray(verdict).astype(str)
    explicit = np.isin(verdict, ["YES", "NO"])
    # UNK may mean uncertainty OR silence. The source does not distinguish them.
    state = np.where(verdict == "YES", "source_positive", np.where(verdict == "NO", "source_negative", "unresolved_or_unmentioned"))
    factor = (1-conflict_scale) + conflict_scale * agreement
    factor *= np.where(explicit, 1.0, 0.25)
    factor *= np.minimum(counts/2, 1)
    return np.clip(base * factor, 0, 1).astype(np.float32), state, agreement.astype(np.float32)
