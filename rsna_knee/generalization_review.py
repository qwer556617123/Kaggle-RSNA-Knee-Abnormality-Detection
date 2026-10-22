"""Paired, source-aware proxy analysis; never estimates an unseen Private score."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .constants import LABELS
from .research_validation import UID, EVAL_COLUMNS, MASK_COLUMNS, masked_auc, paired_group_bootstrap, outer_evaluation_rows


def _macro(truth, predictions, mask):
    values = masked_auc(truth, predictions, mask)
    return float(values.mean()) if np.isfinite(values).all() else None


def review_generalization(frame: pd.DataFrame, control: pd.DataFrame, candidate: pd.DataFrame,
                          *, subgroup_columns=(), trials: int = 1000, bootstrap_samples: int = 2000, seed: int = 20260905) -> dict:
    """Compare fixed, already out-of-fold predictions on identical non-Gold UIDs."""
    if trials < 100:
        raise ValueError('at least 100 fixed 30/70 draws are required')
    expected = frame.iloc[outer_evaluation_rows(frame)].copy()
    if expected[UID].duplicated().any() or expected.report_group.isna().any():
        raise ValueError('evaluation identity or group is invalid')
    for name, table in [('control', control), ('candidate', candidate)]:
        if table[UID].isna().any() or table[UID].duplicated().any() or set(table[UID]) != set(expected[UID]):
            raise ValueError(f'{name} lacks exact non-Gold UID coverage')
        if 'fold' not in table or not table.set_index(UID).loc[expected[UID], 'fold'].eq(expected.fold.to_numpy()).all():
            raise ValueError(f'{name} has an invalid OOF fold assignment')
    expected = expected.reset_index(drop=True)
    a = control.set_index(UID).loc[expected[UID], LABELS].to_numpy(float)
    b = candidate.set_index(UID).loc[expected[UID], LABELS].to_numpy(float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('non-finite OOF prediction')
    truth = expected[EVAL_COLUMNS].to_numpy(float)
    mask = expected[MASK_COLUMNS].to_numpy(bool)
    base = masked_auc(truth, a, mask)
    new = masked_auc(truth, b, mask)
    if not np.isfinite(base).all() or not np.isfinite(new).all():
        raise ValueError('twelve-target support required for the overall comparison')
    paired = paired_group_bootstrap(truth, a, b, mask, expected.report_group, samples=bootstrap_samples, seed=seed)
    fold_deltas = {}
    for fold in (0, 1, 2):
        rows = expected.fold.to_numpy() == fold
        first, second = _macro(truth[rows], a[rows], mask[rows]), _macro(truth[rows], b[rows], mask[rows])
        if first is None or second is None:
            raise ValueError(f'fold {fold} lacks twelve-target support')
        fold_deltas[str(fold)] = second - first
    correlations = {}
    for index, label in enumerate(LABELS):
        correlations[label] = float(pd.Series(a[:, index]).rank().corr(pd.Series(b[:, index]).rank()))
    groups = expected.report_group.astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 10:
        raise ValueError('too few independent groups for a 30/70 stress test')
    rng = np.random.default_rng(seed)
    public_delta, hidden_delta = [], []
    for _ in range(trials):
        shuffled = rng.permutation(unique)
        visible = np.isin(groups, shuffled[:max(1, round(.30 * len(unique)))])
        visible_a, visible_b = _macro(truth[visible], a[visible], mask[visible]), _macro(truth[visible], b[visible], mask[visible])
        hidden_a, hidden_b = _macro(truth[~visible], a[~visible], mask[~visible]), _macro(truth[~visible], b[~visible], mask[~visible])
        if None not in (visible_a, visible_b, hidden_a, hidden_b):
            public_delta.append(visible_b - visible_a)
            hidden_delta.append(hidden_b - hidden_a)
    if len(hidden_delta) < .8 * trials:
        raise ValueError('insufficient class-supported 30/70 simulations')
    visible, hidden = np.asarray(public_delta), np.asarray(hidden_delta)
    subgroup_report = {}
    for column in subgroup_columns:
        if column not in expected:
            raise ValueError(f'missing predeclared subgroup column: {column}')
        subgroup_report[column] = {}
        for value, index in expected.groupby(expected[column].fillna('missing').astype(str)).groups.items():
            rows = np.asarray(list(index), dtype=int)
            supported = []
            per_target = {}
            for j, label in enumerate(LABELS):
                y = truth[rows, j][mask[rows, j]]
                if len(rows) < 50 or (y == 0).sum() < 10 or (y == 1).sum() < 10:
                    continue
                aa = masked_auc(truth[rows, j:j+1], a[rows, j:j+1], mask[rows, j:j+1])[0]
                bb = masked_auc(truth[rows, j:j+1], b[rows, j:j+1], mask[rows, j:j+1])[0]
                per_target[label] = float(bb - aa)
                supported.append(label)
            subgroup_report[column][value] = {'studies': len(rows), 'supported_targets': supported,
                'per_target_delta': per_target,
                'macro_delta': float(np.mean(list(per_target.values()))) if len(supported) == len(LABELS) else None,
                'interpretation': 'comparable' if len(supported) == len(LABELS) else 'descriptive_only'}
    result = {
        'endpoint': 'fixed_proxy_oof_negative_screen', 'private_score_observed': False,
        'private_score_estimated': False, 'gold_excluded': True,
        'studies': len(expected), 'groups': len(unique),
        'control_macro_auc': float(base.mean()), 'candidate_macro_auc': float(new.mean()),
        'macro_delta': float((new - base).mean()),
        'per_target': {label: {'control_auc': float(base[j]), 'candidate_auc': float(new[j]),
                               'delta': float(new[j] - base[j]), 'rank_correlation': correlations[label]}
                       for j, label in enumerate(LABELS)},
        'paired_group_bootstrap_ci95': paired['ci95'], 'fold_deltas': fold_deltas,
        'stress_30_70': {'requested_draws': trials, 'valid_draws': len(hidden),
                         'rank_reversal_fraction': float(np.mean((visible > 0) != (hidden > 0))),
                         'hidden_delta_quantiles_05_50_95': np.quantile(hidden, [.05, .5, .95]).tolist()},
        'subgroups': subgroup_report,
        'limitations': ['Proxy labels derive from report sources.',
                        'Simulated 70% partitions are not the actual Kaggle Private set.',
                        'Patient linkage is unverified unless separately audited.'],
    }
    encoded = json.dumps(result, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
    result['analysis_sha256'] = hashlib.sha256(encoded).hexdigest()
    return result
