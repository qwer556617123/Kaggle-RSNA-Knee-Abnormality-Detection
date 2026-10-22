from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateGate:
    accepted: bool
    gold_gain: float
    weak_gain: float
    reason: str


def evaluate_candidate(
    *,
    candidate_gold_auc: float,
    candidate_weak_auc: float,
    incumbent_gold_auc: float,
    incumbent_weak_auc: float,
    minimum_gold_gain: float = 0.01,
    minimum_weak_gain: float = 0.01,
    maximum_gold_regression: float = 0.02,
) -> CandidateGate:
    gold_gain = candidate_gold_auc - incumbent_gold_auc
    weak_gain = candidate_weak_auc - incumbent_weak_auc
    accepted = gold_gain >= minimum_gold_gain or (
        weak_gain >= minimum_weak_gain and gold_gain >= -maximum_gold_regression
    )
    if gold_gain >= minimum_gold_gain:
        reason = "gold macro AUC gate passed"
    elif weak_gain >= minimum_weak_gain and gold_gain >= -maximum_gold_regression:
        reason = "weak OOF gate passed without excessive gold regression"
    elif gold_gain < -maximum_gold_regression:
        reason = "rejected: gold macro AUC regression exceeds limit"
    else:
        reason = "rejected: neither improvement gate passed"
    return CandidateGate(accepted, gold_gain, weak_gain, reason)
