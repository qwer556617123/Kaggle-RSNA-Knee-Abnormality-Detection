"""Validate an in-memory fixed blend using synthetic IDs and probabilities."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rsna_knee.artifacts import validate_submission
from rsna_knee.constants import LABELS
from rsna_knee.inference_execution import validate_probabilities


def main() -> None:
    rng = np.random.default_rng(42)
    ids = ['synthetic-study-001', 'synthetic-study-002', 'synthetic-study-003']
    reference = pd.DataFrame({'StudyInstanceUID': ids})
    anchor = rng.uniform(0.05, 0.95, (len(ids), len(LABELS)))
    reader = rng.uniform(0.05, 0.95, anchor.shape)
    blended = 0.8 * anchor + 0.2 * reader
    validate_probabilities(blended, public_preview=True)
    submission = pd.DataFrame(blended, columns=LABELS)
    submission.insert(0, 'StudyInstanceUID', ids)
    validate_submission(submission, reference)
    print(json.dumps({'status': 'PASS', 'studies': len(ids), 'targets': len(LABELS),
                      'weights': [0.8, 0.2], 'synthetic_only': True,
                      'files_written': 0, 'network_used': False}))


if __name__ == '__main__':
    main()
