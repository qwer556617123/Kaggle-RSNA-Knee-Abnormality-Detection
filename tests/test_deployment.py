import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rsna_knee.constants import LABELS
from rsna_knee.deployment import (
    append_submission_ledger,
    fallback_limit,
    stage_rank_blend,
    validate_deployment_artifacts,
)


def _bundle(root, *, fallbacks=0, consecutive=0, include_candidate=True):
    rows = ['StudyInstanceUID,' + ','.join(LABELS)]
    for index in range(3):
        rows.append(f'u{index},' + ','.join(str(.1 + .1 * index + .001 * target) for target in range(len(LABELS))))
    submission = root / 'submission.csv'
    submission.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    digest = hashlib.sha256(submission.read_bytes()).hexdigest()
    if include_candidate:
        (root / 'candidate_receipt.json').write_text(json.dumps({
            'candidate_id': 'native-v1', 'candidate_type': 'research', 'actual_branch': 'native',
            'source_sha256': 'a' * 64, 'input_sha256': 'b' * 64,
            'checkpoint_sha256': 'c' * 64, 'checkpoint_verified': True,
            'submission_sha256': digest,
            'kernel_slug': 'example/test', 'kernel_version': 7,
        }))
    (root / 'runtime_receipt.json').write_text(json.dumps({
        'actual_branch': 'native', 'elapsed_seconds': 12.0, 'runtime_upper_seconds': 26999.0,
        'study_count': 3, 'study_fallbacks': fallbacks, 'consecutive_decode_failures': consecutive,
        'runtime_projection_studies': 1500, 'timing_sample_studies': 96,
        'missing_studies': 0, 'test_reports_used': False,
        'fallback_policy': 'forbidden', 'uid_order_verified': True,
    }))


class DeploymentTests(unittest.TestCase):
    def test_bundle_requires_all_root_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _bundle(root, include_candidate=False)
            with self.assertRaises(FileNotFoundError): validate_deployment_artifacts(root)

    def test_bundle_is_bound_to_actual_branch_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _bundle(root)
            report = validate_deployment_artifacts(root)
            self.assertTrue(report['deployment_ready']); self.assertEqual(report['fallback_limit'], 0)
            receipt = json.loads((root / 'candidate_receipt.json').read_text())
            receipt['actual_branch'] = 'staged-but-not-run'
            (root / 'candidate_receipt.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'branches differ'): validate_deployment_artifacts(root)

    def test_fallback_and_consecutive_failures_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _bundle(root, fallbacks=1)
            with self.assertRaisesRegex(ValueError, 'fallback'): validate_deployment_artifacts(root)
            _bundle(root, consecutive=1)
            with self.assertRaisesRegex(ValueError, 'decode failures'): validate_deployment_artifacts(root)
        self.assertEqual(fallback_limit(300), 0); self.assertEqual(fallback_limit(301), 0)

    def test_submission_ledger_prevents_duplicate_kernel_version(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'submission_ledger.jsonl'
            record = {'candidate_id': 'x', 'state': 'submitted', 'kernel_slug': 'example/x', 'kernel_version': 2, 'submission_id': 1}
            append_submission_ledger(ledger, record)
            with self.assertRaisesRegex(ValueError, 'duplicate'): append_submission_ledger(ledger, record)
            self.assertEqual(len(ledger.read_text().splitlines()), 1)

    def test_staged_rank_blend_cannot_be_mistaken_for_submission_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _bundle(root)
            anchor = root / 'submission.csv'; arm = root / 'arm.csv'; arm.write_bytes(anchor.read_bytes())
            staged = root / 'staged'
            manifest = stage_rank_blend(anchor, arm, staged, anchor_weight=.8, candidate_id='x', actual_branch='native')
            self.assertEqual(manifest['status'], 'staged_not_submission_ready')
            self.assertTrue((staged / 'submission.csv').is_file())
            with self.assertRaises(FileNotFoundError): validate_deployment_artifacts(staged)
            with self.assertRaisesRegex(ValueError, 'predeclared 0.8'):
                stage_rank_blend(anchor, arm, staged, anchor_weight=.5, candidate_id='x', actual_branch='native')


if __name__ == '__main__':
    unittest.main()
