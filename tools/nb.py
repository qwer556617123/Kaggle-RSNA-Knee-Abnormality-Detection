"""Build a self-contained Kaggle notebook from a Python kernel entrypoint.

`# %%` separates cells. Shared packages listed in `bundle_packages` inside
kernel-metadata.json are embedded in a bootstrap cell and materialized under
`/kaggle/working/_bundled_source` at runtime.
"""
from __future__ import annotations

import hashlib
import base64
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _cell(source: str, index: int) -> dict:
    return {
        "id": hashlib.sha1(f"{index}:{source[:200]}".encode("utf-8")).hexdigest()[:12],
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": (source.rstrip() + "\n").splitlines(keepends=True),
    }


def _bundle_cell(packages: list[str]) -> str | None:
    files: dict[str, str] = {}
    for package in packages:
        package_root = ROOT / package
        if package_root.is_file() and package_root.suffix == ".py":
            files[package_root.relative_to(ROOT).as_posix()] = package_root.read_text(encoding="utf-8")
            continue
        if not package_root.is_dir():
            raise FileNotFoundError(f"bundle package not found: {package_root}")
        for path in sorted(package_root.rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            files[relative] = path.read_text(encoding="utf-8")
    if not files:
        return None
    serialized = json.dumps(files, ensure_ascii=False)
    return f'''# Materialize repository modules embedded by tools/nb.py.
import json as _bundle_json
import sys as _bundle_sys
from pathlib import Path as _BundlePath

_BUNDLE_ROOT = _BundlePath("/kaggle/working/_bundled_source")
_BUNDLE_FILES = _bundle_json.loads({json.dumps(serialized)})
for _relative, _source in _BUNDLE_FILES.items():
    _target = _BUNDLE_ROOT / _relative
    _target.parent.mkdir(parents=True, exist_ok=True)
    _target.write_text(_source, encoding="utf-8")
if str(_BUNDLE_ROOT) not in _bundle_sys.path:
    _bundle_sys.path.insert(0, str(_BUNDLE_ROOT))
print(f"bundled {{len(_BUNDLE_FILES)}} source files")
'''


def _embed_files_cell(kernel_dir: Path, specs: list[dict]) -> str | None:
    """Embed small immutable binary assets directly in the built notebook."""
    payload: dict[str, dict[str, str]] = {}
    for spec in specs:
        source = (kernel_dir / spec["path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"embedded asset not found: {source}")
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        expected = str(spec["sha256"]).lower()
        if digest != expected:
            raise ValueError(
                f"embedded asset hash mismatch for {source}: {digest} != {expected}"
            )
        name = str(spec.get("name") or source.name)
        if name in payload:
            raise ValueError(f"duplicate embedded asset name: {name}")
        payload[name] = {
            "sha256": digest,
            "base64": base64.b64encode(raw).decode("ascii"),
        }
    if not payload:
        return None
    serialized = json.dumps(payload, separators=(",", ":"))
    return f'''# Materialize immutable assets embedded by tools/nb.py.
import base64 as _embed_base64
import hashlib as _embed_hashlib
import json as _embed_json
from pathlib import Path as _EmbedPath

_EMBED_ROOT = _EmbedPath("/kaggle/working/_embedded_assets")
_EMBED_PAYLOAD = _embed_json.loads({json.dumps(serialized)})
_EMBED_ROOT.mkdir(parents=True, exist_ok=True)
for _name, _record in _EMBED_PAYLOAD.items():
    _raw = _embed_base64.b64decode(_record["base64"], validate=True)
    _digest = _embed_hashlib.sha256(_raw).hexdigest()
    if _digest != _record["sha256"]:
        raise RuntimeError(f"embedded asset {{_name}} hash mismatch")
    (_EMBED_ROOT / _name).write_bytes(_raw)
print(f"materialized {{len(_EMBED_PAYLOAD)}} hash-pinned embedded assets")
'''


def build(kernel_dir: Path) -> Path:
    kernel_dir = kernel_dir.resolve()
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    base_notebook = metadata.get("base_notebook")
    base_source_file = metadata.get("base_source_file")
    if base_notebook and base_source_file:
        raise ValueError("base_notebook and base_source_file are mutually exclusive")
    if base_notebook or base_source_file:
        base_path = (kernel_dir / (base_notebook or base_source_file)).resolve()
        if base_notebook:
            base = json.loads(base_path.read_text(encoding="utf-8"))
            if base.get("nbformat") != 4 or not base.get("cells"):
                raise ValueError(f"invalid base notebook: {base_path}")
            cells = list(base["cells"])
            selected_cells = metadata.get("base_notebook_cells")
            if selected_cells is not None:
                if not isinstance(selected_cells, list) or not selected_cells:
                    raise ValueError("base_notebook_cells must be a nonempty list")
                audited_cells = []
                seen_indices: set[int] = set()
                for selected in selected_cells:
                    if not isinstance(selected, dict):
                        raise ValueError("base_notebook_cells entries must be objects")
                    index = int(selected["index"])
                    if index < 0 or index >= len(cells):
                        raise IndexError(f"notebook cell index out of range: {base_path}:{index}")
                    if index in seen_indices:
                        raise ValueError(f"duplicate base notebook cell index: {index}")
                    seen_indices.add(index)
                    cell = json.loads(json.dumps(cells[index]))
                    source = cell.get("source", "")
                    source_text = "".join(source) if isinstance(source, list) else str(source)
                    digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
                    expected = str(selected["sha256"]).lower()
                    if digest != expected:
                        raise ValueError(
                            f"base notebook cell hash mismatch for {base_path}:{index}: "
                            f"{digest} != {expected}"
                        )
                    cell["outputs"] = []
                    cell["execution_count"] = None
                    audited_cells.append(cell)
                cells = audited_cells
        else:
            cells = [_cell(base_path.read_text(encoding="utf-8"), 0)]
        replacements = metadata.get("base_source_replacements", {})
        if replacements:
            if not isinstance(replacements, dict):
                raise ValueError("base_source_replacements must be an object")
            for old, new in replacements.items():
                if not isinstance(old, str) or not old:
                    raise ValueError("base source replacement keys must be nonempty strings")
                count = 0
                for index, cell in enumerate(cells):
                    if cell.get("cell_type") != "code":
                        continue
                    source = cell.get("source", "")
                    is_list = isinstance(source, list)
                    text = "".join(source) if is_list else str(source)
                    hits = text.count(old)
                    if hits:
                        text = text.replace(old, str(new))
                        cell = dict(cell)
                        cell["source"] = text.splitlines(keepends=True) if is_list else text
                        cells[index] = cell
                        count += hits
                if count != 1:
                    raise ValueError(
                        f"base source replacement must match exactly once, got {count}: {old!r}"
                    )

        append_notebooks = metadata.get("append_notebook_cells", [])
        if not isinstance(append_notebooks, list):
            raise ValueError("append_notebook_cells must be a list")
        for spec in append_notebooks:
            source_path = (kernel_dir / spec["path"]).resolve()
            source_notebook = json.loads(source_path.read_text(encoding="utf-8"))
            source_cells = source_notebook.get("cells", [])
            for selected in spec.get("cells", []):
                index = int(selected["index"])
                if index < 0 or index >= len(source_cells):
                    raise IndexError(f"notebook cell index out of range: {source_path}:{index}")
                cell = json.loads(json.dumps(source_cells[index]))
                source = cell.get("source", "")
                source_text = "".join(source) if isinstance(source, list) else str(source)
                digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
                expected = str(selected["sha256"]).lower()
                if digest != expected:
                    raise ValueError(
                        f"appended notebook cell hash mismatch for {source_path}:{index}: "
                        f"{digest} != {expected}"
                    )
                cell["outputs"] = []
                cell["execution_count"] = None
                cells.append(cell)
        append_file = metadata.get("append_source_file")
        if not append_file:
            raise ValueError("base source requires append_source_file")
        embedded = _embed_files_cell(kernel_dir, list(metadata.get("embed_files", [])))
        if embedded:
            cells.append(_cell(embedded, len(cells)))
        bootstrap = _bundle_cell(list(metadata.get("bundle_packages", [])))
        if bootstrap:
            cells.append(_cell(bootstrap, len(cells)))
        append_path = (kernel_dir / append_file).resolve()
        text = append_path.read_text(encoding="utf-8")
        marker = metadata.get("append_source_after")
        if marker:
            offset = text.find(str(marker))
            if offset < 0:
                raise ValueError(f"append marker not found in {append_path}: {marker!r}")
            text = text[offset:]
        blocks = [block.strip("\n") for block in text.split("\n# %%\n") if block.strip()]
        cells.extend(_cell(block, len(cells) + index) for index, block in enumerate(blocks))
    else:
        code_file = metadata.get("source_file")
        if code_file:
            source_path = kernel_dir / code_file
        else:
            sources = sorted(kernel_dir.glob("*.py"))
            if not sources:
                raise SystemExit(f"no .py entrypoint in {kernel_dir}")
            source_path = sources[0]

        text = source_path.read_text(encoding="utf-8")
        prelude = metadata.get("source_prelude")
        if prelude:
            text = str(prelude).rstrip() + "\n" + text
        blocks = [block.strip("\n") for block in text.split("\n# %%\n") if block.strip()]
        bootstrap = _bundle_cell(list(metadata.get("bundle_packages", [])))
        if bootstrap:
            blocks.insert(0, bootstrap)
        embedded = _embed_files_cell(kernel_dir, list(metadata.get("embed_files", [])))
        if embedded:
            blocks.insert(0, embedded)
        cells = [_cell(block, index) for index, block in enumerate(blocks)]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_name = metadata.get("code_file")
    if not output_name:
        output_name = f"{(base_path if (base_notebook or base_source_file) else source_path).stem}.ipynb"
    output = kernel_dir / output_name
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return output


if __name__ == "__main__":
    print(build(Path(sys.argv[1])))
