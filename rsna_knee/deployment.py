"""Fail-closed deployment artifacts and an append-only submission ledger.

Kernel completion is only a compute event.  A candidate becomes deployable
only after its exact root-level CSV, candidate receipt and runtime receipt
validate together.  This module deliberately has no Kaggle API dependency so
the same checks run locally and in an offline kernel.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .constants import LABELS


REQUIRED_CANDIDATE_FIELDS = frozenset({
    'candidate_id', 'candidate_type', 'actual_branch', 'source_sha256',
    'input_sha256', 'checkpoint_sha256', 'checkpoint_verified',
    'submission_sha256', 'kernel_slug', 'kernel_version',
})
REQUIRED_RUNTIME_FIELDS = frozenset({
    'actual_branch', 'elapsed_seconds', 'study_count', 'study_fallbacks',
    'consecutive_decode_failures', 'runtime_upper_seconds',
    'runtime_projection_studies', 'timing_sample_studies', 'missing_studies',
    'test_reports_used', 'fallback_policy', 'uid_order_verified',
})
LEDGER_STATES = frozenset({
    'built', 'running', 'completed', 'evaluated', 'submission_ready',
    'submitted', 'scored', 'deployment_failed', 'execution_error', 'retired',
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def fallback_limit(study_count: int) -> int:
    if isinstance(study_count, bool) or not isinstance(study_count, int) or study_count < 1:
        raise ValueError('study_count must be a positive integer')
    return 0


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'missing required {label}: {path.name}')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise ValueError(f'invalid {label} JSON') from error
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be an object')
    return value


def _validate_csv(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError('missing required submission.csv')
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        expected = ['StudyInstanceUID', *LABELS]
        if reader.fieldnames != expected:
            raise ValueError(f'submission schema mismatch: {reader.fieldnames}')
        rows = list(reader)
    if len(rows) < 3:
        raise ValueError('submission contains too few studies')
    uids = [row['StudyInstanceUID'] for row in rows]
    if any(not uid for uid in uids) or len(set(uids)) != len(uids):
        raise ValueError('submission has empty or duplicate UIDs')
    values = np.asarray([[float(row[label]) for label in LABELS] for row in rows], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('submission predictions must be finite and in [0, 1]')
    # A constant target is not a usable candidate, even if other targets vary.
    if any(np.unique(values[:, column]).size == 1 for column in range(values.shape[1])):
        raise ValueError('submission is constant for at least one target')
    return {'rows': len(rows), 'sha256': sha256_file(path)}


def validate_deployment_artifacts(output_dir: str | Path, *, max_runtime_seconds: float = 7.5 * 3600) -> dict:
    """Validate an offline kernel's final deployment bundle.

    No inferred defaults are allowed: a receipt must name the exact branch,
    bind the CSV hash, and record actual rather than permitted fallbacks.
    """
    root = Path(output_dir)
    csv_info = _validate_csv(root / 'submission.csv')
    candidate = _load_json(root / 'candidate_receipt.json', 'candidate receipt')
    runtime = _load_json(root / 'runtime_receipt.json', 'runtime receipt')
    missing_candidate = REQUIRED_CANDIDATE_FIELDS - set(candidate)
    missing_runtime = REQUIRED_RUNTIME_FIELDS - set(runtime)
    if missing_candidate or missing_runtime:
        raise ValueError(f'missing receipt fields: candidate={sorted(missing_candidate)}, runtime={sorted(missing_runtime)}')
    if candidate['submission_sha256'] != csv_info['sha256']:
        raise ValueError('candidate receipt does not bind submission CSV hash')
    for field in ('source_sha256', 'input_sha256', 'checkpoint_sha256'):
        value = candidate[field]
        if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
            raise ValueError(f'candidate receipt has invalid {field}')
    if candidate['checkpoint_verified'] is not True:
        raise ValueError('candidate checkpoint hash was not verified')
    if candidate['actual_branch'] != runtime['actual_branch']:
        raise ValueError('candidate/runtime actual branches differ')
    count = runtime['study_count']
    if isinstance(count, bool) or not isinstance(count, int) or count != csv_info['rows']:
        raise ValueError('runtime study_count does not match submission rows')
    for field in ('elapsed_seconds', 'runtime_upper_seconds'):
        if not isinstance(runtime[field], (int, float)) or not np.isfinite(runtime[field]) or runtime[field] < 0:
            raise ValueError(f'invalid runtime field: {field}')
    if runtime['runtime_projection_studies'] != 1500:
        raise ValueError('runtime upper bound must project exactly 1,500 studies')
    if (isinstance(runtime['timing_sample_studies'], bool)
            or not isinstance(runtime['timing_sample_studies'], int)
            or runtime['timing_sample_studies'] < 96):
        raise ValueError('runtime timing sample must contain at least 96 training studies')
    if runtime['runtime_upper_seconds'] > max_runtime_seconds:
        raise ValueError('runtime upper bound exceeds 7.5-hour deployment limit')
    fallbacks = runtime['study_fallbacks']
    consecutive = runtime['consecutive_decode_failures']
    if isinstance(fallbacks, bool) or not isinstance(fallbacks, int) or fallbacks < 0:
        raise ValueError('invalid study_fallbacks')
    if isinstance(consecutive, bool) or not isinstance(consecutive, int) or consecutive < 0:
        raise ValueError('invalid consecutive_decode_failures')
    if fallbacks != 0 or runtime['fallback_policy'] != 'forbidden':
        raise ValueError('all study fallbacks are forbidden')
    if consecutive != 0:
        raise ValueError('decode failures must fail closed')
    if runtime['missing_studies'] != 0:
        raise ValueError('deployment may not omit a study')
    if runtime['test_reports_used'] is not False:
        raise ValueError('test reports are forbidden')
    if runtime['uid_order_verified'] is not True:
        raise ValueError('test UID order was not verified')
    return {
        'deployment_ready': True,
        'submission_sha256': csv_info['sha256'],
        'study_count': count,
        'fallback_limit': 0,
        'runtime_projection_studies': 1500,
        'timing_sample_studies': runtime['timing_sample_studies'],
        'actual_branch': candidate['actual_branch'],
        'candidate_id': candidate['candidate_id'],
        'kernel_slug': candidate['kernel_slug'],
        'kernel_version': candidate['kernel_version'],
    }


def append_submission_ledger(path: str | Path, record: dict) -> dict:
    """Append a validated state transition; never overwrite prior evidence."""
    required = {'candidate_id', 'state', 'kernel_slug', 'kernel_version'}
    missing = required - set(record)
    if missing:
        raise ValueError(f'ledger record missing {sorted(missing)}')
    if record['state'] not in LEDGER_STATES:
        raise ValueError('invalid ledger state')
    destination = Path(path)
    existing = []
    if destination.is_file():
        existing = [json.loads(line) for line in destination.read_text(encoding='utf-8').splitlines() if line.strip()]
    key = (str(record.get('competition', 'rsna-knee-abnormality-detection')), str(record['kernel_slug']), int(record['kernel_version']))
    if record['state'] == 'submitted' and any(
        (str(row.get('competition', 'rsna-knee-abnormality-detection')), str(row.get('kernel_slug')), int(row.get('kernel_version', -1))) == key
        and row.get('state') == 'submitted' for row in existing
    ):
        raise ValueError('duplicate formal submission kernel/version')
    entry = dict(record)
    entry.setdefault('competition', 'rsna-knee-abnormality-detection')
    entry.setdefault('recorded_at_utc', datetime.now(timezone.utc).isoformat())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('a', encoding='utf-8', newline='') as handle:
        handle.write(json.dumps(entry, sort_keys=True) + '\n')
    return entry


def stage_probability_blend(anchor_csv, reader_csv, output_dir, *, recipe_lock, recipe_lock_sha256,
                            expected_reader_sha256, reader_public_score_at=None):
    """Recompute the locked80/20 probabilities; source/deployment admission stays separate."""
    import pandas as pd
    from datetime import datetime
    lock_path=Path(recipe_lock)
    if sha256_file(lock_path)!=recipe_lock_sha256:raise ValueError('blend recipe lock hash mismatch')
    locked=_load_json(lock_path,'blend lock')
    if (locked.get('anchor_weight')!=.8 or locked.get('candidate_weight')!=.2
            or locked.get('locked_before_reader_public_score') is not True or 'per_target_weights' in locked):
        raise ValueError('only the prelocked global80/20 probability blend is permitted')
    timestamp=datetime.fromisoformat(locked['locked_at_utc'].replace('Z','+00:00'))
    if timestamp.tzinfo is None:raise ValueError('blend lock timestamp must have timezone')
    if reader_public_score_at is not None and timestamp>=datetime.fromisoformat(reader_public_score_at.replace('Z','+00:00')):
        raise ValueError('blend lock must precede reader Public score')
    if sha256_file(anchor_csv)!=locked['anchor_csv_sha256'] or sha256_file(reader_csv)!=expected_reader_sha256:
        raise ValueError('blend source CSV hash mismatch')
    _validate_csv(Path(anchor_csv));_validate_csv(Path(reader_csv))
    a=pd.read_csv(anchor_csv,dtype={'StudyInstanceUID':str});b=pd.read_csv(reader_csv,dtype={'StudyInstanceUID':str})
    if not a.StudyInstanceUID.equals(b.StudyInstanceUID):raise ValueError('blend inputs require identical UID order')
    combined=a.copy();combined[LABELS]=.8*a[LABELS].to_numpy(np.float64)+.2*b[LABELS].to_numpy(np.float64)
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    output=root/'submission.csv';combined.to_csv(output,index=False,float_format='%.17g')
    result=dict(anchor_csv_sha256=sha256_file(anchor_csv),reader_csv_sha256=sha256_file(reader_csv),
        submission_sha256=sha256_file(output),recipe_lock_sha256=recipe_lock_sha256,
        anchor_weight=.8,candidate_weight=.2,targets=LABELS,uid_order_verified=True,
        deployment_eligible=False,formal_submission_ready=False)
    (root/'probability_blend_receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def stage_rank_blend(anchor_csv: str | Path, arm_csv: str | Path, output_dir: str | Path, *,
                     anchor_weight: float, candidate_id: str, actual_branch: str) -> dict:
    """Create a traceable rank blend, but never claim it is submission-ready.

    The caller must separately add the real candidate/runtime receipts after the
    complete inference branch has run. This prevents a completed partial arm
    from being mistaken for a final blend.
    """
    if not isinstance(anchor_weight, (int, float)) or not np.isfinite(anchor_weight) or abs(anchor_weight - .8) > 1e-12:
        raise ValueError('anchor_weight must equal the predeclared 0.8')
    import pandas as pd
    anchor_path, arm_path, root = Path(anchor_csv), Path(arm_csv), Path(output_dir)
    anchor = pd.read_csv(anchor_path, dtype={'StudyInstanceUID': str})
    arm = pd.read_csv(arm_path, dtype={'StudyInstanceUID': str})
    expected = ['StudyInstanceUID', *LABELS]
    if list(anchor.columns) != expected or list(arm.columns) != expected:
        raise ValueError('blend inputs have incompatible schema')
    if not anchor['StudyInstanceUID'].equals(arm['StudyInstanceUID']):
        raise ValueError('blend inputs have UID/order mismatch')
    values = []
    for label in LABELS:
        left = anchor[label].to_numpy(float); right = arm[label].to_numpy(float)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError('blend inputs have non-finite values')
        values.append(anchor_weight * pd.Series(left).rank(method='average', pct=True).to_numpy()
                      + (1-anchor_weight) * pd.Series(right).rank(method='average', pct=True).to_numpy())
    result = pd.DataFrame(np.stack(values, axis=1), columns=LABELS)
    result.insert(0, 'StudyInstanceUID', anchor['StudyInstanceUID'])
    root.mkdir(parents=True, exist_ok=True)
    result.to_csv(root / 'submission.csv', index=False)
    manifest = {
        'status': 'staged_not_submission_ready', 'candidate_id': candidate_id,
        'actual_branch': actual_branch, 'anchor_weight': float(anchor_weight),
        'anchor_submission_sha256': sha256_file(anchor_path), 'arm_submission_sha256': sha256_file(arm_path),
        'submission_sha256': sha256_file(root / 'submission.csv'),
        'required_followup': ['candidate_receipt.json', 'runtime_receipt.json', 'validate_deployment_artifacts'],
    }
    (root / 'stage_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    return manifest
