"""Fail-closed schema and optional reference validation for Kaggle outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def validate(output_dir: Path, expected_sha256: str | None = None) -> dict:
    submission_path = output_dir.resolve() / "submission.csv"
    if not submission_path.is_file():
        raise FileNotFoundError(submission_path)
    digest = hashlib.sha256(submission_path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise ValueError(f"submission SHA-256 mismatch: {digest}")
    with submission_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["StudyInstanceUID", *TARGETS]:
            raise ValueError(f"submission schema mismatch: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) < 3:
        raise ValueError(f"submission has too few studies: {len(rows)}")
    ids = [row["StudyInstanceUID"] for row in rows]
    if any(not uid for uid in ids) or len(ids) != len(set(ids)):
        raise ValueError("submission study IDs are empty or duplicated")
    values = []
    for row in rows:
        target_values = [float(row[target]) for target in TARGETS]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in target_values):
            raise ValueError("submission contains non-finite or out-of-range predictions")
        values.extend(target_values)
    if len(set(values)) == 1:
        raise ValueError("submission is globally constant")
    return {"status": "PASS", "studies": len(rows), "sha256": digest}


if __name__ == "__main__":
    if len(sys.argv) not in {2, 3}:
        raise SystemExit("usage: validate_submission_output.py OUTPUT_DIR [EXPECTED_SHA256]")
    print(json.dumps(validate(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) == 3 else None), sort_keys=True))
