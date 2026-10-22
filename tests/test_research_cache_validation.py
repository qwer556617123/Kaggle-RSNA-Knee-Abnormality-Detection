import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rsna_knee.research_cache_validation import validate_cache


class ResearchCacheValidationTests(unittest.TestCase):
    def make_cache(self, root, *, arrays=True):
        uids = ['u0', 'u1']
        index = {uid: {'shard': 0, 'row': i} for i, uid in enumerate(uids)}
        (root/'cache_index.json').write_text(json.dumps(index), encoding='utf-8')
        records = []
        for i, uid in enumerate(uids):
            records.append(json.dumps({
                'StudyInstanceUID': uid, 'row': i, 'shard': 0,
                'window_count': 24,
            }))
        (root/'metadata_000.jsonl').write_text('\n'.join(records) + '\n', encoding='utf-8')
        (root/'failures_000.jsonl').write_text('', encoding='utf-8')
        if arrays:
            np.save(root/'images_000.npy', np.zeros((2,24,3,336,336), dtype=np.uint8))
            np.save(root/'mask_000.npy', np.ones((2,24), dtype=np.float32))
        (root/'montage_000.png').write_bytes(b'png')
        (root/'montage_manifest.json').write_text('{}', encoding='utf-8')
        (root/'cache_receipt.json').write_text(json.dumps({
            'status': 'complete', 'study_count': 2, 'all_studies_retained': True,
            'study_fallback_count': 0, 'failure_count': 0,
            'shards': [{'shard': 0, 'rows': 2}],
            'runtime_receipt': {'study_seconds': [1.0, 1.0]},
        }), encoding='utf-8')

    def test_valid_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_cache(root)
            result = validate_cache(root, expected_studies=2)
            self.assertEqual(result['status'], 'passed')

    def test_missing_arrays_fail_when_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_cache(root, arrays=False)
            with self.assertRaises(ValueError):
                validate_cache(root)
            self.assertEqual(validate_cache(root, require_arrays=False)['status'], 'passed')

    def test_nonzero_failure_records_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_cache(root)
            (root/'failures_000.jsonl').write_text('{}\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                validate_cache(root)

    def test_reviewed_series_rejection_requires_consistent_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_cache(root, arrays=False)
            index = {f'u{i}': {'shard': 0, 'row': i} for i in range(20)}
            failure = {'StudyInstanceUID': 'u0', 'SeriesInstanceUID': 'rejected',
                       'slot': 'Axial_Fluid', 'error': 'ValueError: declared plane differs from slice normal by over 30 degrees'}
            records = [{'StudyInstanceUID': uid, 'row': i, 'window_count': 24,
                        'metadata': [{'slot': 'Axial_Fluid', 'SeriesInstanceUID': 'accepted'}],
                        'failures': [failure] if i == 0 else []} for i, uid in enumerate(index)]
            (root/'cache_index.json').write_text(json.dumps(index))
            (root/'metadata_000.jsonl').write_text('\n'.join(map(json.dumps, records)))
            (root/'failures_000.jsonl').write_text(json.dumps(failure))
            receipt = json.loads((root/'cache_receipt.json').read_text())
            receipt.update(study_count=20, failure_count=1, shards=[{'shard': 0, 'rows': 20}],
                           runtime_receipt={'study_seconds': [1.] * 20})
            (root/'cache_receipt.json').write_text(json.dumps(receipt))
            self.assertEqual(validate_cache(root, require_arrays=False)['affected_studies'], 1)
            records[0]['metadata'][0]['SeriesInstanceUID'] = 'rejected'
            (root/'metadata_000.jsonl').write_text('\n'.join(map(json.dumps, records)))
            with self.assertRaisesRegex(ValueError, 'rejected series reused'):
                validate_cache(root, require_arrays=False)


if __name__ == '__main__':
    unittest.main()
