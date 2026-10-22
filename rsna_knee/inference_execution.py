"""Separate measured preview work from competition inference; no model fallback."""
import os
from pathlib import Path


def competition_rerun():
    return bool(os.environ.get('KAGGLE_IS_COMPETITION_RERUN'))


def execution_splits(rerun):
    return ('test',) if rerun else ('train', 'test')


def competition_root(input_root):
    input_root = Path(input_root)
    candidates = [input_root / 'rsna-knee-abnormality-detection',
                  input_root / 'competitions/rsna-knee-abnormality-detection']
    present = [root for root in candidates if (root / 'test.csv').is_file() and (root / 'test_series.csv').is_file()]
    if len(present) != 1:
        raise ValueError('one unambiguous competition test mount required')
    return present[0]


def validate_probabilities(values, *, public_preview):
    import numpy as np
    minimum = 3 if public_preview else 1
    if values.ndim != 2 or values.shape[1] != 12 or len(values) < minimum:
        raise ValueError('invalid test UID/target shape')
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('nonfinite or out-of-range probabilities')
    # A one-row output has zero range by definition, independently of quality.
    # The actual3+96 preview still enforces all12 targets are nonconstant.
    if len(values) >= 3 and (np.ptp(values, axis=0) == 0).any():
        raise ValueError('constant target predictions')
