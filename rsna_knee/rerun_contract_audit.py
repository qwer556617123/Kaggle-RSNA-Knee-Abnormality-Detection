"""Independently recompute test-only probe identity from downloaded CSV bytes."""
import hashlib
import json
from pathlib import Path


def audit_rerun_contract(output_dir, manifest):
    import numpy as np
    import pandas as pd
    from .constants import LABELS
    output = Path(output_dir)
    reference = pd.read_csv(output / 'submission.csv', dtype={'StudyInstanceUID':str})
    report = json.loads((output.parent / 'rerun_contract/receipt.json').read_text(encoding='utf-8'))
    if report.get('status') != 'COMPLETE' or report.get('actual_hidden_data_accessed') is not False:
        raise ValueError('test-only public MRI evidence is not complete')
    if [case['test_count'] for case in report['cases']] != [1, 2, 3]:
        raise ValueError('real one/two/three-study contract cases required')
    for case in report['cases']:
        count = case['test_count']
        root = output.parent / 'rerun_contract' / str(count) / 'native_reader_inference'
        receipt_path, csv_path = root / 'receipt.json', root / 'submission.csv'
        if (hashlib.sha256(receipt_path.read_bytes()).hexdigest() != case['receipt_sha256']
                or hashlib.sha256(csv_path.read_bytes()).hexdigest() != case['csv_sha256']):
            raise ValueError('test-only receipt/CSV hash differs')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        actual = pd.read_csv(csv_path, dtype={'StudyInstanceUID':str})
        if (list(actual.columns) != ['StudyInstanceUID', *LABELS]
                or actual.StudyInstanceUID.tolist() != reference.StudyInstanceUID.tolist()[:count]):
            raise ValueError('test-only output UID/order/target mismatch')
        values = actual[LABELS].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('test-only probabilities invalid')
        maximum = float(np.max(np.abs(values - reference[LABELS].to_numpy(float)[:count])))
        if maximum > 1e-6 or abs(maximum - case['maximum_probability_difference']) > 1e-8:
            raise ValueError('test-only predictions differ from preview; cannot trust reported delta')
        if (receipt.get('status') != 'COMPLETE' or receipt.get('execution_mode') != 'competition_rerun'
                or receipt.get('runner_sha256') != manifest['code_sha256']
                or receipt.get('training_timing_study_count') != 0 or receipt.get('test_study_count') != count
                or receipt.get('decode_fallback_count') != 0 or receipt.get('missing_study_count') != 0
                or receipt.get('test_reports_read') is not False or case.get('test_only_mount') is not True
                or case.get('training_access_attempts') != 0):
            raise ValueError('test-only execution/scope contract invalid')
        for name in case['dataframe_read_paths']:
            path = Path(name)
            if path.name in ('train.csv', 'train_series.csv') or 'train_series' in path.parts:
                raise ValueError('competition inference reads training files')
    return {'cases':3, 'public_mri_predictions':6, 'actual_hidden_evaluation':False,
            'maximum_allowed_probability_difference':1e-6}
