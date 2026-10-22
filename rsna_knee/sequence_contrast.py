"""Competition-MRI-only pixel summaries for a CPU sequence-contrast pilot.

The research cache has 24 windows in five fixed slots (4, 6, 6, 4, 4).
No external encoder, report text, or learned image feature is used here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

COUNTS = (4, 6, 6, 4, 4)
PAIRS = ((1, 3), (2, 4))  # sagittal and coronal Fluid / NonFluid
SUMMARY_DIM = 8
FEATURE_DIM = 5 * SUMMARY_DIM + 5 + 2 * SUMMARY_DIM + 2


def uid_sha256(uids) -> str:
    return hashlib.sha256("\n".join(map(str, uids)).encode()).hexdigest()


def pixel_summary(window: np.ndarray) -> np.ndarray:
    """Eight robust statistics on a fixed 8x spatial subsample of 3 slices."""
    if window.shape != (3, 336, 336) or window.dtype != np.uint8:
        raise ValueError("expected raw 3x336x336 uint8 cache window")
    x = window[:, ::8, ::8].astype(np.float32) / 255.0
    center = x[:, 14:28, 14:28]
    border = np.concatenate((x[:, :7].ravel(), x[:, -7:].ravel(),
                             x[:, 7:-7, :7].ravel(), x[:, 7:-7, -7:].ravel()))
    return np.asarray((x.mean(), x.std(), *np.quantile(x, (.1, .5, .9)),
                       center.mean(), border.mean(),
                       (np.abs(np.diff(x, axis=1)).mean() + np.abs(np.diff(x, axis=2)).mean()) / 2),
                      dtype=np.float32)


def study_summaries(images: np.ndarray, mask: np.ndarray):
    if images.shape != (24, 3, 336, 336) or mask.shape != (24,):
        raise ValueError("cache image/mask shape mismatch")
    if not np.isin(mask, (0, 1)).all() or not mask.any():
        raise ValueError("invalid cache mask")
    slots = np.zeros((5, SUMMARY_DIM), dtype=np.float32)
    pseudo = np.zeros((2, SUMMARY_DIM), dtype=np.float32)
    present = np.zeros(5, dtype=np.float32)
    offset = 0
    for slot, count in enumerate(COUNTS):
        indices = [i for i in range(offset, offset + count) if mask[i]]
        if indices:
            vectors = np.stack([pixel_summary(images[i]) for i in indices])
            slots[slot] = vectors.mean(0)
            present[slot] = 1
            if slot in (1, 2) and len(vectors) > 1:
                half = len(vectors) // 2
                pseudo[slot - 1] = np.abs(vectors[:half].mean(0) - vectors[half:].mean(0))
        offset += count
    return slots, pseudo, present


def design_matrix(slots: np.ndarray, pseudo: np.ndarray, present: np.ndarray,
                  *, arm: str, groups=None, seed=20260905) -> np.ndarray:
    """Same-dimensional control/candidate/shuffle input; no label use.

    Shuffling is restricted to supplied partition groups, so a diagnostic can
    never pull a train study into held-out feature construction.
    """
    slots, pseudo, present = map(np.asarray, (slots, pseudo, present))
    n = len(slots)
    if slots.shape != (n, 5, SUMMARY_DIM) or pseudo.shape != (n, 2, SUMMARY_DIM) or present.shape != (n, 5):
        raise ValueError("summary array shapes differ")
    if arm not in {"control", "contrast", "shuffled"}:
        raise ValueError("unknown sequence arm")
    contrast = np.zeros((n, 2, SUMMARY_DIM), dtype=np.float32)
    pair_mask = np.zeros((n, 2), dtype=np.float32)
    rng = np.random.default_rng(seed)
    groups = np.zeros(n, dtype=int) if groups is None else np.asarray(groups)
    if len(groups) != n:
        raise ValueError("partition-group length mismatch")
    for pair_index, (fluid, nonfluid) in enumerate(PAIRS):
        eligible = (present[:, fluid] > 0) & (present[:, nonfluid] > 0)
        pair_mask[:, pair_index] = eligible
        if arm == "control":
            contrast[eligible, pair_index] = pseudo[eligible, pair_index]
        elif arm == "contrast":
            contrast[eligible, pair_index] = np.abs(slots[eligible, fluid] - slots[eligible, nonfluid])
        else:
            for group in np.unique(groups):
                rows = np.flatnonzero(eligible & (groups == group))
                if len(rows) < 2:
                    pair_mask[rows, pair_index] = 0
                    continue
                # A derangement prevents accidentally retaining the real pairing.
                shift = int(rng.integers(1, len(rows)))
                donors = np.roll(rows, shift)
                contrast[rows, pair_index] = np.abs(slots[rows, fluid] - slots[donors, nonfluid])
    result = np.concatenate((slots.reshape(n, -1), present, contrast.reshape(n, -1), pair_mask), axis=1)
    if result.shape != (n, FEATURE_DIM) or not np.isfinite(result).all():
        raise ValueError("invalid sequence-contrast design matrix")
    return result.astype(np.float32)


def load_summary_collection(cache_roots, expected_uids):
    """Read only indexed research-cache pixels, binding UID order exactly."""
    expected_uids = list(map(str, expected_uids))
    if len(set(expected_uids)) != len(expected_uids):
        raise ValueError("duplicate expected UID")
    locations = {}
    for root in map(Path, cache_roots):
        index = json.loads((root / "cache_index.json").read_text(encoding="utf-8"))
        for uid, location in index.items():
            if uid in locations:
                raise ValueError("duplicate cache UID")
            locations[uid] = (root, location)
    if set(locations) != set(expected_uids):
        raise ValueError("cache UID coverage differs from frozen cohort")
    arrays = {}
    slots = np.zeros((len(expected_uids), 5, SUMMARY_DIM), np.float32)
    pseudo = np.zeros((len(expected_uids), 2, SUMMARY_DIM), np.float32)
    present = np.zeros((len(expected_uids), 5), np.float32)
    for i, uid in enumerate(expected_uids):
        root, location = locations[uid]
        key = (root, int(location["shard"]))
        if key not in arrays:
            shard = key[1]
            arrays[key] = (np.load(root / f"images_{shard:03d}.npy", mmap_mode="r"),
                           np.load(root / f"mask_{shard:03d}.npy", mmap_mode="r"))
        images, masks = arrays[key]
        row = int(location["row"])
        slots[i], pseudo[i], present[i] = study_summaries(images[row], masks[row])
    return slots, pseudo, present
