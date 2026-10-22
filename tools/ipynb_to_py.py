"""Extract code cells from a Jupyter notebook without executing it."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def extract(source: Path, destination: Path) -> None:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    blocks: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        value = cell.get("source", "")
        text = "".join(value) if isinstance(value, list) else str(value)
        if text.strip():
            blocks.append(text.rstrip())
    if not blocks:
        raise ValueError(f"notebook contains no code cells: {source}")
    destination.write_text("\n\n# %%\n\n".join(blocks) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: ipynb_to_py.py SOURCE.ipynb DESTINATION.py")
    extract(Path(sys.argv[1]), Path(sys.argv[2]))
