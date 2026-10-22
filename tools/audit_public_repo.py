"""Audit tracked public files and optional Git history without printing secrets."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

PRIVATE_DIRS = {
    '.private_archive', '.git', 'research_state', 'research', 'data', 'assets',
    'output', 'output_versions', 'deployment_input', 'artifacts', 'cache',
    '.kaggle', '.venv', 'venv', '__pycache__', '.pytest_cache',
}
PRIVATE_SUFFIXES = {
    '.dcm', '.dicom', '.csv', '.tsv', '.parquet', '.feather', '.pt', '.pth',
    '.ckpt', '.safetensors', '.onnx', '.pb', '.npy', '.npz', '.pkl', '.pickle',
    '.whl', '.zip', '.tar', '.gz', '.bundle', '.log', '.pem', '.key', '.p12', '.pfx',
    '.pyc', '.exe', '.dll', '.pyd',
}
PATTERNS = {
    'private_key': re.compile(r'-{5}BEGIN (?:RSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-{5}'),
    'access_token': re.compile(r'\b(?:KGAT_[A-Za-z0-9_=-]{16,}|hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9_-]{30,})'),
    'credential_value': re.compile(r'''(?i)["'](?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password)["']\s*[:=]\s*["'][^"'\s]{16,}["']'''),
    'credential_assignment': re.compile(r'''(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password)\s*=\s*["'][^"'\s]{16,}["']'''),
    'kaggle_credential_pair': re.compile(r'''(?is)["']username["']\s*:\s*["'][^"']+["'].{0,160}["']key["']\s*:\s*["'][^"'\s]{16,}["']'''),
    'bearer_value': re.compile(r'(?i)\bBearer\s+[A-Za-z0-9_.=-]{20,}'),
    'windows_user_path': re.compile(r'(?i)[A-Z]:[\\/]+Users[\\/]+'),
    'unix_user_path': re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/'),
    'project_account': re.compile(r'qwer\d{6,}', re.I),
}
MAX_BYTES = 1_000_000


def inspect_file(name: str, data: bytes) -> list[dict]:
    """Return finding categories and line numbers, never matched values."""
    path = Path(name)
    findings = []
    if any(part in PRIVATE_DIRS or part.startswith('.tmp') for part in path.parts):
        findings.append({'path': name, 'kind': 'private_directory'})
    if path.suffix.lower() in PRIVATE_SUFFIXES or path.name == 'kaggle.json' or path.name.startswith('.env'):
        findings.append({'path': name, 'kind': 'private_file_type'})
    if len(data) > MAX_BYTES:
        findings.append({'path': name, 'kind': 'oversized_public_file'})
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        return findings + [{'path': name, 'kind': 'unexpected_binary'}]
    for kind, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({'path': name, 'kind': kind, 'line': text.count('\n', 0, match.start()) + 1})
    if path.suffix == '.ipynb':
        # The public distribution excludes generated notebooks entirely.
        findings.append({'path': name, 'kind': 'notebook_not_allowed'})
        try:
            notebook = json.loads(text)
            for cell in notebook.get('cells', []):
                if cell.get('outputs') or cell.get('execution_count') is not None or cell.get('attachments'):
                    findings.append({'path': name, 'kind': 'notebook_execution_or_attachment'})
                    break
        except (json.JSONDecodeError, AttributeError):
            findings.append({'path': name, 'kind': 'invalid_notebook'})
    return findings


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ['git', '-c', f'safe.directory={root.as_posix()}', *args], cwd=root,
        stderr=subprocess.PIPE,
    )


def audit(root: Path, *, history: bool = False) -> dict:
    root = root.resolve()
    names = [p.decode('utf-8') for p in git(root, 'ls-files', '-z').split(b'\0') if p]
    if not names:
        raise ValueError('No tracked public files to audit')
    findings = []
    # Inspect both the staged blobs and current files: either can contain a leak.
    for name in names:
        findings.extend(inspect_file(name, git(root, 'show', f':{name}')))
        path = root / name
        if path.is_symlink():
            findings.append({'path': name, 'kind': 'symlink_not_allowed'})
        elif not path.is_file():
            findings.append({'path': name, 'kind': 'tracked_file_missing'})
        else:
            findings.extend(inspect_file(name, path.read_bytes()))
    objects = 0
    if history:
        entries = git(root, 'rev-list', '--objects', '--all').decode('utf-8').splitlines()
        for entry in entries:
            oid, _, name = entry.partition(' ')
            if git(root, 'cat-file', '-t', oid).strip() != b'blob':
                continue
            objects += 1
            for finding in inspect_file(name, git(root, 'cat-file', 'blob', oid)):
                findings.append({**finding, 'object': oid[:12]})
    # Keep duplicate findings from staged/current scans out of the report.
    unique = {json.dumps(f, sort_keys=True): f for f in findings}
    return {'status': 'PASS' if not unique else 'FAIL', 'tracked_files': len(names),
            'historical_blobs': objects, 'findings': list(unique.values())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--history', action='store_true')
    args = parser.parse_args(argv)
    report = audit(args.root, history=args.history)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
