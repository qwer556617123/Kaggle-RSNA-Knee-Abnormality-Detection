"""Deterministic evidence-state audit for report-derived supervision.

States describe whether the tested multilingual rule system found evidence;
they are not imaging truth and they do not alter target scores by themselves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


VALID_STATES = frozenset({'explicit_positive', 'explicit_negative', 'silent'})


def state_from_counts(n_positive, n_negative):
    pos = np.asarray(n_positive)
    neg = np.asarray(n_negative)
    if pos.shape != neg.shape or not np.isfinite(pos).all() or not np.isfinite(neg).all() or (pos < 0).any() or (neg < 0).any():
        raise ValueError('invalid evidence counts')
    # Positive evidence takes precedence over a contradictory normality clause;
    # conflicts remain visible through the separate conflict flag.
    return np.where(pos > 0, 'explicit_positive', np.where(neg > 0, 'explicit_negative', 'silent'))


def state_summary(states, verdicts):
    states = np.asarray(states, dtype=object)
    verdicts = np.asarray(verdicts, dtype=object).astype(str)
    if states.shape != verdicts.shape or not set(np.unique(states)).issubset(VALID_STATES):
        raise ValueError('invalid states or verdicts')
    explicit = np.isin(verdicts, ['YES', 'NO'])
    expected = np.where(verdicts == 'YES', 'explicit_positive', 'explicit_negative')
    agreement = (states == expected) & explicit
    return {
        'rows': int(len(states)),
        'state_counts': {name: int((states == name).sum()) for name in sorted(VALID_STATES)},
        'source_explicit_rows': int(explicit.sum()),
        'source_explicit_agreement': float(agreement[explicit].mean()) if explicit.any() else None,
        'rule_source_conflict_rows': int(((states == 'explicit_positive') & (verdicts == 'NO') | (states == 'explicit_negative') & (verdicts == 'YES')).sum()),
    }
