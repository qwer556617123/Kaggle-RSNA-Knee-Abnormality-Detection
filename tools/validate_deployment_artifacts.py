"""Fail-closed root-output deployment validator for Kaggle kernels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rsna_knee.deployment import validate_deployment_artifacts


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 1:
        raise SystemExit('usage: validate_deployment_artifacts.py OUTPUT_DIR')
    report = validate_deployment_artifacts(Path(argv[0]))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
