"""Reallocate missing-slot quotas to real windows, preserving the 12-index budget."""
import numpy as np

POOLS = (4, 6, 6, 4, 4)
COUNTS = (2, 3, 3, 2, 2)


def select_valid_training_windows(mask, rng=np.random):
    mask = np.asarray(mask)
    if mask.shape != (24,) or not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError('expected 24 binary window flags with real MRI coverage')
    mask = mask.astype(bool)
    selected, offset = [], 0
    for pool, count in zip(POOLS, COUNTS):
        selected.extend((offset + np.sort(rng.choice(pool, count, replace=False))).tolist())
        offset += pool
    # Full coverage reproduces the parent choices and RNG state exactly.
    if mask.all():
        return np.asarray(selected, dtype=np.int64)
    valid = set(i for i in selected if mask[i])
    target = min(12, int(mask.sum()))
    offsets = np.cumsum((0, *POOLS))
    # Round robin across available slots prevents a larger slot swallowing the quota.
    remaining = [list(rng.permutation([i for i in range(offsets[s], offsets[s+1])
                                      if mask[i] and i not in valid])) for s in range(5)]
    while len(valid) < target:
        for choices in remaining:
            if choices and len(valid) < target:
                valid.add(int(choices.pop()))
    # Sparse studies keep masked placeholders for collate shape; never duplicate real windows.
    padding = [i for i in selected if not mask[i]][:12-len(valid)]
    result = np.asarray(sorted(valid.union(padding)), dtype=np.int64)
    if len(result) != 12 or len(set(result)) != 12 or mask[result].sum() != target:
        raise RuntimeError('valid-window allocation invariant failed')
    return result
