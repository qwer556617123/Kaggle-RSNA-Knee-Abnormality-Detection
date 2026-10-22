from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator


PRUNED_INPUT_DIRECTORIES = {"train_series", "test_series", ".git", "__pycache__"}


def iter_files_pruned(root: str | Path, filename: str | None = None) -> Iterator[Path]:
    """Walk Kaggle inputs without descending into the 800k-file DICOM trees."""
    root = Path(root)
    if not root.exists():
        return
    for directory, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in PRUNED_INPUT_DIRECTORIES]
        if filename is not None:
            if filename in files:
                yield Path(directory) / filename
        else:
            for name in files:
                yield Path(directory) / name
