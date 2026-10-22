"""Synthetic research-contract fixture; never competition or medical evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.constants import CONFIDENCE_COLUMNS, LABELS
from rsna_knee.research_validation import EVAL_COLUMNS, MASK_COLUMNS, UID, nested_partition, table_digest


def write_synthetic_contract(root: Path, count: int = 180) -> None:
    rows = np.arange(count)
    evaluation = pd.DataFrame({
        UID: [f'synthetic-{i:04d}' for i in rows],
        'report_group': [f'synthetic-group-{i // 2:03d}' for i in rows],
        'fold': (rows // 2) % 3,
        'is_gold': (rows == 0).astype(int),
    })
    for index, name in enumerate(EVAL_COLUMNS):
        evaluation[name] = (rows + index) % 2
    for name in MASK_COLUMNS:
        evaluation[name] = True
    control = evaluation[[UID, 'report_group']].copy()
    for index, name in enumerate(LABELS):
        control[name] = np.where((rows + index) % 2, 0.8, 0.2)
    for name in CONFIDENCE_COLUMNS:
        control[name] = 0.8
    candidate = control.copy()
    candidate[CONFIDENCE_COLUMNS] = 0.4
    tables = {'evaluation.csv': evaluation, 'labels_control.csv': control, 'labels_evidence.csv': candidate}
    restored = {}
    for name, frame in tables.items():
        frame.to_csv(root / name, index=False, float_format='%.9g')
        restored[name] = pd.read_csv(root / name, dtype={UID: str})
    evaluation = restored['evaluation.csv']
    manifest = {'contract_version': 'top10-clean-v1',
                'evaluation_sha256': table_digest(evaluation),
                'label_control_sha256': table_digest(restored['labels_control.csv']),
                'label_evidence_sha256': table_digest(restored['labels_evidence.csv']),
                'fold_sha256': table_digest(evaluation[[UID, 'report_group', 'fold']]),
                'nested_splits': {}, 'scope': 'SYNTHETIC TEST FIXTURE ONLY'}
    for fold in range(3):
        partition = nested_partition(evaluation, fold)
        manifest['nested_splits'][str(fold)] = {}
        for name in ('train', 'inner', 'outer', 'excluded'):
            ids = evaluation.iloc[getattr(partition, name)][UID]
            manifest['nested_splits'][str(fold)][name] = {
                'count': len(ids), 'uid_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
            }
    (root / 'research_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
