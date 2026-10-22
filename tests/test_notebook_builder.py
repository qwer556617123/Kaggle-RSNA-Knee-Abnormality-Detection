import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.nb import build


class NotebookCompositionTests(unittest.TestCase):
    def test_selects_hash_pinned_base_notebook_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = ["FIRST = 1\n", "SECOND = 2\n", "THIRD = 3\n"]
            (root / "base.ipynb").write_text(
                json.dumps({
                    "cells": [
                        {
                            "cell_type": "code", "source": [source], "metadata": {},
                            "outputs": [{"old": True}], "execution_count": index + 1,
                        }
                        for index, source in enumerate(sources)
                    ],
                    "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
                }),
                encoding="utf-8",
            )
            (root / "append.py").write_text("TAIL = 4\n", encoding="utf-8")
            (root / "kernel-metadata.json").write_text(
                json.dumps({
                    "code_file": "composed.ipynb",
                    "base_notebook": "base.ipynb",
                    "base_notebook_cells": [
                        {
                            "index": index,
                            "sha256": hashlib.sha256(sources[index].encode()).hexdigest(),
                        }
                        for index in (0, 2)
                    ],
                    "append_source_file": "append.py",
                }),
                encoding="utf-8",
            )

            built = json.loads(build(root).read_text(encoding="utf-8"))
            built_sources = ["".join(cell["source"]) for cell in built["cells"]]
            self.assertEqual(built_sources, [sources[0], sources[2], "TAIL = 4\n"])
            self.assertEqual(built["cells"][0]["outputs"], [])
            self.assertIsNone(built["cells"][0]["execution_count"])

    def test_rejects_base_notebook_cell_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.ipynb").write_text(
                json.dumps({
                    "cells": [{
                        "cell_type": "code", "source": ["BASE = 1\n"],
                        "metadata": {}, "outputs": [], "execution_count": None,
                    }],
                    "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
                }),
                encoding="utf-8",
            )
            (root / "append.py").write_text("TAIL = 2\n", encoding="utf-8")
            (root / "kernel-metadata.json").write_text(
                json.dumps({
                    "code_file": "composed.ipynb",
                    "base_notebook": "base.ipynb",
                    "base_notebook_cells": [{"index": 0, "sha256": "0" * 64}],
                    "append_source_file": "append.py",
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "base notebook cell hash mismatch"):
                build(root)
    def test_bundles_packages_before_appended_source_for_base_notebook(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            kernel = root / "kernel"
            kernel.mkdir()
            (kernel / "base.ipynb").write_text(
                json.dumps({
                    "cells": [{
                        "cell_type": "code", "execution_count": None,
                        "metadata": {}, "outputs": [], "source": ["BASE = True\n"],
                    }],
                    "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
                }),
                encoding="utf-8",
            )
            (kernel / "append.py").write_text(
                "from rsna_knee import __name__\n", encoding="utf-8"
            )
            (kernel / "kernel-metadata.json").write_text(
                json.dumps({
                    "code_file": "composed.ipynb",
                    "base_notebook": "base.ipynb",
                    "append_source_file": "append.py",
                    "bundle_packages": ["rsna_knee"],
                }),
                encoding="utf-8",
            )

            notebook = json.loads(build(kernel).read_text(encoding="utf-8"))
            sources = ["".join(cell["source"]) for cell in notebook["cells"]]
            bundle_index = next(i for i, source in enumerate(sources) if "_BUNDLE_FILES" in source)
            append_index = next(i for i, source in enumerate(sources) if "from rsna_knee import __name__" in source)
            self.assertLess(bundle_index, append_index)

    def test_applies_exactly_one_audited_base_source_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.ipynb").write_text(
                json.dumps({
                    "cells": [{"cell_type": "code", "source": "ASSET = 'old'\n", "metadata": {}, "outputs": [], "execution_count": None}],
                    "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
                }),
                encoding="utf-8",
            )
            (root / "append.py").write_text("TAIL = 2\n", encoding="utf-8")
            (root / "kernel-metadata.json").write_text(
                json.dumps({
                    "code_file": "composed.ipynb",
                    "base_notebook": "base.ipynb",
                    "append_source_file": "append.py",
                    "base_source_replacements": {"ASSET = 'old'": "ASSET = 'dynamic'"},
                }),
                encoding="utf-8",
            )
            notebook = json.loads(build(root).read_text(encoding="utf-8"))
            self.assertEqual(notebook["cells"][0]["source"], "ASSET = 'dynamic'\n")

    def test_appends_marked_source_to_base_notebook(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.ipynb"
            base.write_text(
                json.dumps({
                    "cells": [{"cell_type": "code", "source": ["BASE = 1\n"], "metadata": {}, "outputs": [], "execution_count": None}],
                    "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
                }),
                encoding="utf-8",
            )
            (root / "append.py").write_text(
                "ignored = True\nMARKER = 1\n# %%\nTAIL = 2\n", encoding="utf-8"
            )
            (root / "kernel-metadata.json").write_text(
                json.dumps({
                    "code_file": "composed.ipynb",
                    "base_notebook": "base.ipynb",
                    "append_source_file": "append.py",
                    "append_source_after": "MARKER = 1",
                }),
                encoding="utf-8",
            )
            output = build(root)
            notebook = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(notebook["cells"]), 3)
            sources = ["".join(cell["source"]) for cell in notebook["cells"]]
            self.assertEqual(sources[0], "BASE = 1\n")
            self.assertTrue(sources[1].startswith("MARKER = 1"))
            self.assertNotIn("ignored", sources[1])
            self.assertIn("TAIL = 2", sources[2])

    def test_appends_hash_pinned_public_notebook_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebook = {
                "cells": [{"cell_type": "code", "source": ["BASE = 1\n"], "metadata": {}, "outputs": [], "execution_count": None}],
                "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
            }
            (root / "base.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
            public_source = "PUBLIC_BRANCH = 2\n"
            public = dict(notebook)
            public["cells"] = [{"cell_type": "code", "source": [public_source], "metadata": {}, "outputs": [{"old": True}], "execution_count": 1}]
            (root / "public.ipynb").write_text(json.dumps(public), encoding="utf-8")
            (root / "append.py").write_text("TAIL = 3\n", encoding="utf-8")
            (root / "kernel-metadata.json").write_text(
                json.dumps({
                    "code_file": "composed.ipynb", "base_notebook": "base.ipynb",
                    "append_source_file": "append.py",
                    "append_notebook_cells": [{"path": "public.ipynb", "cells": [{
                        "index": 0, "sha256": hashlib.sha256(public_source.encode()).hexdigest(),
                    }]}],
                }),
                encoding="utf-8",
            )
            built = json.loads(build(root).read_text(encoding="utf-8"))
            sources = ["".join(cell["source"]) for cell in built["cells"]]
            self.assertEqual(sources, ["BASE = 1\n", public_source, "TAIL = 3\n"])
    def test_embeds_hash_pinned_asset_before_appended_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = b"model-bytes\x00\xff"
            (root / "asset.bin").write_bytes(raw)
            (root / "base.ipynb").write_text(
                json.dumps({
                    "cells": [{"cell_type": "code", "source": ["BASE = 1\n"], "metadata": {}, "outputs": [], "execution_count": None}],
                    "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
                }),
                encoding="utf-8",
            )
            (root / "append.py").write_text("TAIL = 2\n", encoding="utf-8")
            (root / "kernel-metadata.json").write_text(
                json.dumps({
                    "code_file": "composed.ipynb",
                    "base_notebook": "base.ipynb",
                    "append_source_file": "append.py",
                    "embed_files": [{
                        "path": "asset.bin",
                        "name": "pinned.bin",
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }],
                }),
                encoding="utf-8",
            )
            notebook = json.loads(build(root).read_text(encoding="utf-8"))
            sources = ["".join(cell["source"]) for cell in notebook["cells"]]
            self.assertEqual(len(sources), 3)
            self.assertIn("_embedded_assets", sources[1])
            self.assertIn("pinned.bin", sources[1])
            self.assertIn("TAIL = 2", sources[2])

    def test_rejects_embedded_asset_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "asset.bin").write_bytes(b"drift")
            (root / "base.ipynb").write_text(
                json.dumps({
                    "cells": [{"cell_type": "code", "source": ["BASE = 1\n"], "metadata": {}, "outputs": [], "execution_count": None}],
                    "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
                }),
                encoding="utf-8",
            )
            (root / "append.py").write_text("TAIL = 2\n", encoding="utf-8")
            (root / "kernel-metadata.json").write_text(
                json.dumps({
                    "code_file": "composed.ipynb",
                    "base_notebook": "base.ipynb",
                    "append_source_file": "append.py",
                    "embed_files": [{
                        "path": "asset.bin",
                        "sha256": "0" * 64,
                    }],
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build(root)

    def test_composes_audited_base_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "public.py").write_text("ROOT = 'old'\nRUN = True\n", encoding="utf-8")
            (root / "validate.py").write_text("VALID = True\n", encoding="utf-8")
            (root / "kernel-metadata.json").write_text(
                json.dumps({
                    "code_file": "composed.ipynb",
                    "base_source_file": "public.py",
                    "append_source_file": "validate.py",
                    "base_source_replacements": {"ROOT = 'old'": "ROOT = 'dynamic'"},
                }),
                encoding="utf-8",
            )
            notebook = json.loads(build(root).read_text(encoding="utf-8"))
            sources = ["".join(cell["source"]) for cell in notebook["cells"]]
            self.assertEqual(sources, ["ROOT = 'dynamic'\nRUN = True\n", "VALID = True\n"])

if __name__ == "__main__":
    unittest.main()
