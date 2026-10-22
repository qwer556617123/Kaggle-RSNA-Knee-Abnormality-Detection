import json
import subprocess

from tools.audit_public_repo import audit, inspect_file


def test_secret_is_reported_without_disclosing_value():
    secret = 'hf_' + 'x' * 32
    report = inspect_file('settings.txt', ('token=' + secret).encode())
    assert any(f['kind'] == 'access_token' for f in report)
    assert secret not in json.dumps(report)


def test_generic_assignment_and_kaggle_credential_pair_are_rejected():
    value = 'x' * 32
    assignment = inspect_file('settings.py', ('API_KEY = "' + value + '"').encode())
    pair = inspect_file('renamed.json', json.dumps({'username': 'synthetic', 'key': value}).encode())
    assert any(f['kind'] == 'credential_assignment' for f in assignment)
    assert any(f['kind'] == 'kaggle_credential_pair' for f in pair)


def test_medical_artifacts_and_notebook_outputs_are_rejected():
    assert inspect_file('data/sample.dcm', b'not-real-dicom')
    notebook = {'cells': [{'outputs': [{'text': ['private output']}], 'execution_count': 1}]}
    report = inspect_file('example.ipynb', json.dumps(notebook).encode())
    assert any(f['kind'] == 'notebook_execution_or_attachment' for f in report)


def test_history_scan_detects_secret_removed_from_worktree(tmp_path):
    def git(*args):
        return subprocess.run(['git', '-c', f'safe.directory={tmp_path.as_posix()}', *args],
                              cwd=tmp_path, check=True, capture_output=True)
    git('init', '-b', 'main')
    git('config', 'user.name', 'Synthetic Test')
    git('config', 'user.email', 'test@example.invalid')
    path = tmp_path / 'settings.txt'
    path.write_text('hf_' + 'x' * 32, encoding='utf-8')
    git('add', 'settings.txt')
    git('-c', 'commit.gpgsign=false', 'commit', '-m', 'Synthetic fixture')
    path.write_text('clean', encoding='utf-8')
    git('add', 'settings.txt')
    assert audit(tmp_path)['status'] == 'PASS'
    report = audit(tmp_path, history=True)
    assert report['status'] == 'FAIL'
    assert any(f.get('object') for f in report['findings'])


def test_current_unstaged_secret_is_detected(tmp_path):
    git = ['git', '-c', f'safe.directory={tmp_path.as_posix()}']
    subprocess.run(git + ['init', '-b', 'main'], cwd=tmp_path, check=True, capture_output=True)
    path = tmp_path / 'settings.txt'
    path.write_text('clean', encoding='utf-8')
    subprocess.run(git + ['add', 'settings.txt'], cwd=tmp_path, check=True, capture_output=True)
    path.write_text('hf_' + 'x' * 32, encoding='utf-8')
    assert audit(tmp_path)['status'] == 'FAIL'
