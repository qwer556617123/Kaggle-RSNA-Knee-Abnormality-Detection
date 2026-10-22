from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import LABELS


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "unversioned")


def experiment_manifest(
    experiment_id: str,
    pipeline: str,
    config: Mapping[str, Any],
    *,
    started_at: float,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "pipeline": pipeline,
        "revision": git_revision(),
        "started_unix": started_at,
        "finished_unix": time.time(),
        "runtime_seconds": time.time() - started_at,
        "python": platform.python_version(),
        "config": dict(config),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_json(path: str | Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def validate_submission(
    submission: pd.DataFrame,
    test: pd.DataFrame,
    labels: Sequence[str] = LABELS,
) -> pd.DataFrame:
    expected = ["StudyInstanceUID", *labels]
    if list(submission.columns) != expected:
        raise ValueError(f"submission columns must be exactly {expected}")
    if len(submission) != len(test):
        raise ValueError(f"submission has {len(submission)} rows; expected {len(test)}")
    if submission["StudyInstanceUID"].duplicated().any():
        raise ValueError("submission contains duplicate StudyInstanceUID values")
    if submission["StudyInstanceUID"].tolist() != test["StudyInstanceUID"].tolist():
        raise ValueError("submission order/UIDs do not match test.csv")
    values = submission[list(labels)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("submission contains NaN or infinite predictions")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("submission predictions must be in [0, 1]")
    return submission


def rank_columns(values: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(np.asarray(values, dtype=np.float64))
    return frame.rank(method="average", pct=True).to_numpy(dtype=np.float32)


def validate_feature_artifact(
    features: np.ndarray,
    mask: np.ndarray,
    ids: Sequence[str],
    expected_ids: Sequence[str],
    *,
    slots: int = 5,
    slices: int = 9,
    chunk_rows: int = 128,
) -> int:
    """Fail closed on incomplete or misaligned frozen-feature artifacts."""
    expected = [str(value) for value in expected_ids]
    observed = [str(value) for value in ids]
    if observed != expected:
        raise ValueError("frozen feature IDs do not align exactly with the expected study order")
    if len(set(observed)) != len(observed):
        raise ValueError("frozen feature IDs contain duplicates")
    if features.ndim != 4 or features.shape[:3] != (len(expected), slots, slices):
        raise ValueError(f"unexpected frozen feature shape {features.shape}")
    if features.shape[3] <= 0 or not np.issubdtype(features.dtype, np.floating):
        raise ValueError(f"invalid frozen feature dtype/dimension {features.dtype} {features.shape}")
    if mask.shape != (len(expected), slots):
        raise ValueError(f"unexpected frozen feature mask shape {mask.shape}")
    if not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError("frozen feature mask must be finite and binary")
    if (mask.sum(axis=1) == 0).any():
        raise ValueError("frozen feature artifact contains a study with no decoded protocol slot")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    for start in range(0, len(expected), chunk_rows):
        value = np.asarray(features[start:start + chunk_rows])
        if not np.isfinite(value).all():
            raise ValueError(f"frozen features contain NaN/inf near row {start}")
    return int(features.shape[3])
