"""Fail-closed, UID-aligned comparisons on the frozen research evaluation truth."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .constants import LABELS
from .research_coatnet import PRETRAIN_SHA
from .research_validation import (
    UID, EVAL_COLUMNS, MASK_COLUMNS, nested_partition, masked_auc,
    paired_group_bootstrap, acceptance_gate,
)
from .research_contract import validate_comparison


def uid_hash(values):
    return hashlib.sha256('\n'.join(map(str,values)).encode()).hexdigest()


def validate_oof(frame, prediction, run, contract):
    """Run manifest must trace all three clean-init folds, not just cite old OOF."""
    for key in ('contract_version','evaluation_sha256','fold_sha256'):
        if run.get(key) != contract[key]:
            raise ValueError(f'OOF contract differs: {key}')
    if run.get('pretrain_sha256') != PRETRAIN_SHA or run.get('checkpoint_selection') != 'inner_only':
        raise ValueError('OOF needs approved clean initialization and inner-only selection')
    if run.get('gold_role') != 'excluded_with_report_groups':
        raise ValueError('Gold must be excluded from this cohort')
    for field in ('recipe_sha256', 'preprocessing_sha256', 'code_version'):
        if not run.get(field):
            raise ValueError(f'missing provenance: {field}')
    expected_rows = np.concatenate([nested_partition(frame,f).outer for f in range(3)])
    expected = frame.iloc[np.sort(expected_rows)]
    if prediction[UID].isna().any() or prediction[UID].duplicated().any():
        raise ValueError('invalid prediction UIDs')
    if set(prediction[UID]) != set(expected[UID]) or len(prediction) != len(expected):
        raise ValueError('incomplete OOF or excluded studies included')
    aligned = prediction.set_index(UID).loc[expected[UID]].reset_index()
    if not np.array_equal(aligned.fold.to_numpy(),expected.fold.to_numpy()):
        raise ValueError('prediction folds differ from frozen folds')
    values = aligned[LABELS].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('invalid OOF probabilities')
    if not np.any(np.ptp(values,axis=0) > 0):
        raise ValueError('constant OOF output')
    if set(run.get('folds',{})) != {'0','1','2'}:
        raise ValueError('complete three-fold provenance required')
    for fold in range(3):
        partition = nested_partition(frame,fold)
        provenance = run['folds'][str(fold)]
        for name in ('train','inner','outer','excluded'):
            if provenance.get(name+'_uid_sha256') != uid_hash(frame.iloc[getattr(partition,name)][UID]):
                raise ValueError(f'fold {fold} {name} provenance mismatch')
        if not provenance.get('checkpoint_sha256') or provenance.get('decode_failures') != 0:
            raise ValueError('checkpoint and zero-failure receipt required')
        if provenance.get('selected_epoch',0) < 1:
            raise ValueError('missing inner-selected epoch')
    return expected.reset_index(drop=True), values


def compare_oof(frame, control_pred, candidate_pred, control_run, candidate_run, contract, *, samples=2000):
    comparison = validate_comparison(control_run, candidate_run)
    expected, control = validate_oof(frame,control_pred,control_run,contract)
    other, candidate = validate_oof(frame,candidate_pred,candidate_run,contract)
    if not expected[UID].equals(other[UID]):
        raise ValueError('OOF UID alignment failed')
    truth = expected[EVAL_COLUMNS].to_numpy(float)
    mask = expected[MASK_COLUMNS].to_numpy(bool)
    report = paired_group_bootstrap(truth,control,candidate,mask,expected.report_group,samples=samples)
    fold_deltas, folds = [], {}
    for fold in range(3):
        rows = expected.fold.eq(fold).to_numpy()
        a, b = masked_auc(truth[rows],control[rows],mask[rows]), masked_auc(truth[rows],candidate[rows],mask[rows])
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('each fold needs twelve-target support')
        delta = float((b-a).mean())
        fold_deltas.append(delta)
        folds[str(fold)] = {'control_auc':dict(zip(LABELS,a.tolist())), 'candidate_auc':dict(zip(LABELS,b.tolist())), 'delta':delta}
    report['folds'] = folds
    report['coverage'] = dict(zip(LABELS,mask.sum(axis=0).tolist()))
    report['reference_label_caveat'] = contract['evaluation_label_caveat']
    report['comparison'] = comparison
    report['statistical_gate'] = acceptance_gate(report,fold_deltas,complete_oof=True,compatible=True)
    # Statistical support alone is not enough to submit: runtime and external
    # subgroup receipts are separate mandatory gates, never default to passed.
    report['formal_submission_ready'] = False
    report['remaining_gates'] = ['representative_end_to_end_runtime', 'language_protocol_source_subgroups', 'official_test_smoke', 'submission_quota']
    return report
