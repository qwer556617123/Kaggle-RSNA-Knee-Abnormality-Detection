"""Frozen evaluation, nested report-group splits and research acceptance gates.

This contract is deliberately separate from legacy experiments. Gold58 and all
reports sharing their fingerprints are development references, never selection
or training data in the new research cohort.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .constants import LABELS

UID = "StudyInstanceUID"
EVAL_COLUMNS = [f"{label}__eval" for label in LABELS]
MASK_COLUMNS = [f"{label}__eval_mask" for label in LABELS]


def table_digest(frame: pd.DataFrame) -> str:
    """Hash values and row order with a platform-independent representation."""
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n", float_format="%.9g").encode()).hexdigest()


@dataclass(frozen=True)
class NestedPartition:
    train: np.ndarray
    inner: np.ndarray
    outer: np.ndarray
    excluded: np.ndarray


def nested_partition(frame: pd.DataFrame, fold: int, seed: int = 20260905) -> NestedPartition:
    if fold not in (0, 1, 2):
        raise ValueError("outer fold must be 0, 1, or 2")
    if frame[UID].duplicated().any() or frame[[UID, "report_group", "fold"]].isna().any().any():
        raise ValueError("invalid contract identities")
    if not frame.fold.isin([0, 1, 2]).all() or not frame.is_gold.isin([0, 1, False, True]).all():
        raise ValueError("invalid fold or Gold indicator")
    groups = frame.report_group.astype(str)
    if frame.groupby("report_group").fold.nunique().max() != 1:
        raise ValueError("report group crosses outer folds")
    gold_groups = set(groups[frame.is_gold.astype(bool)])
    excluded = groups.isin(gold_groups).to_numpy()
    outer = frame.fold.to_numpy() == fold
    # Membership depends only on group identity and outer-fold ID, not labels.
    inner = groups.map(lambda g: int(hashlib.sha256(f"{seed}:{fold}:{g}".encode()).hexdigest()[:8], 16) / 2**32 < 0.20).to_numpy()
    partition = NestedPartition(
        np.flatnonzero(~excluded & ~outer & ~inner),
        np.flatnonzero(~excluded & ~outer & inner),
        np.flatnonzero(~excluded & outer),
        np.flatnonzero(excluded),
    )
    if any(len(x) == 0 for x in (partition.train, partition.inner, partition.outer)):
        raise ValueError("empty nested split")
    sets = [set(groups.iloc[x]) for x in (partition.train, partition.inner, partition.outer, partition.excluded)]
    if any(sets[i] & sets[j] for i in range(4) for j in range(i)):
        raise ValueError("report group leakage in nested partition")
    return partition


def outer_evaluation_rows(frame: pd.DataFrame) -> np.ndarray:
    """Frozen non-Gold cohort, excluding all Gold report-group relatives."""
    if frame[UID].duplicated().any() or frame[[UID, "report_group", "fold"]].isna().any().any():
        raise ValueError("invalid evaluation identities")
    gold_groups = set(frame.loc[frame.is_gold.astype(bool), "report_group"].astype(str))
    return np.flatnonzero(~frame.report_group.astype(str).isin(gold_groups).to_numpy())

def masked_auc(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    truth, prediction, mask = np.asarray(truth), np.asarray(prediction), np.asarray(mask)
    if not np.isin(mask, [0, 1, False, True]).all():
        raise ValueError("invalid evaluation mask")
    mask = mask.astype(bool)
    if truth.ndim != 2 or truth.shape != prediction.shape or mask.shape != truth.shape:
        raise ValueError("evaluation shapes differ")
    if not np.isfinite(prediction).all() or not np.isfinite(truth[mask]).all():
        raise ValueError("nonfinite evaluation input")
    result = np.full(truth.shape[1], np.nan)
    for j in range(truth.shape[1]):
        y = truth[mask[:, j], j]
        if not np.isin(y, [0, 1]).all():
            raise ValueError("evaluation truth must be binary")
        if len(np.unique(y)) == 2:
            result[j] = roc_auc_score(y, prediction[mask[:, j], j])
    return result


def paired_group_bootstrap(truth, control, candidate, mask, groups, *, samples=2000, seed=20260905):
    """Draw whole report groups, keeping paired model predictions together."""
    truth, control, candidate, mask = map(np.asarray, (truth, control, candidate, mask))
    if truth.ndim != 2 or truth.shape[1] != len(LABELS):
        raise ValueError("expected twelve targets")
    if pd.isna(groups).any():
        raise ValueError("missing report group")
    groups = np.asarray(groups).astype(str)
    if len(groups) != len(truth):
        raise ValueError("bootstrap groups do not align")
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    if len(members) < 2 or samples < 1:
        raise ValueError("bootstrap requires multiple groups and positive samples")
    base = masked_auc(truth, control, mask)
    new = masked_auc(truth, candidate, mask)
    supported = np.isfinite(base) & np.isfinite(new)
    if not supported.all():
        raise ValueError("all 12 targets require class support for macro acceptance")
    rng, draws = np.random.default_rng(seed), []
    for _ in range(samples):
        rows = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        a = masked_auc(truth[rows], control[rows], mask[rows])
        b = masked_auc(truth[rows], candidate[rows], mask[rows])
        if np.isfinite(a).all() and np.isfinite(b).all():
            draws.append(float((b - a).mean()))
    if len(draws) < samples * 0.8:
        raise ValueError("insufficient bootstrap draws with class support")
    return {"macro_delta": float((new - base).mean()), "per_label_delta": dict(zip(LABELS, (new-base).tolist())),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(), "valid_draws": len(draws),
            "bootstrap_unit": "report_group", "gold_role": "historical_development_reference"}


def acceptance_gate(report: dict, fold_deltas: list[float], *, complete_oof: bool, compatible: bool) -> dict:
    if len(report["ci95"]) != 2 or not np.isfinite([report["macro_delta"], *report["ci95"], *fold_deltas]).all() or report["ci95"][0] > report["ci95"][1]:
        raise ValueError("invalid gate metrics")
    eligible = (complete_oof and compatible and len(fold_deltas) == 3
                and sum(x > 0 for x in fold_deltas) >= 2 and report["macro_delta"] >= 0.003)
    supported = eligible and report["ci95"][0] > 0
    return {"eligible": bool(eligible), "status": "supported" if supported else "exploratory" if eligible else "rejected",
            "formal_submission_ready": bool(supported), "fold_deltas": fold_deltas}


def runtime_upper_bound(fixed_seconds: float, study_seconds: list[float], *, hidden_studies=1300,
                        observed_sizes=(48, 96), seed=20260905) -> dict:
    times = np.asarray(study_seconds, dtype=float)
    if tuple(observed_sizes) != (48, 96) or len(times) < 96:
        raise ValueError("both 48- and 96-study benchmarks are required")
    if hidden_studies < 1 or not np.isfinite(times).all() or (times <= 0).any() or not np.isfinite(fixed_seconds) or fixed_seconds < 0:
        raise ValueError("invalid runtime measurements")
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(times, size=(2000, len(times)), replace=True), axis=1)
    rate = max(float(np.quantile(means, 0.95)), float(times[:48].mean()), float(times[48:96].mean()))
    upper = fixed_seconds + hidden_studies * rate * 1.15
    return {"fixed_seconds": fixed_seconds, "hidden_studies_scenario": hidden_studies,
            "projected_upper_seconds": upper, "passes_7_5h": bool(upper <= 27000),
            "study_seconds_mean": float(times.mean()), "uncertainty_margin": 1.15}
