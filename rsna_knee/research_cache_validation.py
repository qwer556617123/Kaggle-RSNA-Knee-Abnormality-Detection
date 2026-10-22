"""Fail-closed validation for raw-DICOM research cache artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_cache(root, *, expected_studies=None, require_arrays=True):
    """Validate receipt, index, metadata and shard shapes before training."""
    root = Path(root)
    receipt_path = root / 'cache_receipt.json'
    index_path = root / 'cache_index.json'
    if not receipt_path.is_file() or not index_path.is_file():
        raise ValueError('cache receipt/index missing')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('status') != 'complete' or not receipt.get('all_studies_retained'):
        raise ValueError('cache receipt is not complete and all-retained')
    if receipt.get('study_fallback_count', 1) != 0:
        raise ValueError('cache contains study fallback')
    study_count = int(receipt.get('study_count', 0))
    if expected_studies is not None and study_count != int(expected_studies):
        raise ValueError(f'unexpected cache study count {study_count} != {expected_studies}')
    runtime = receipt.get('runtime_receipt') or {}
    study_seconds = runtime.get('study_seconds')
    if (not isinstance(study_seconds, list) or len(study_seconds) != study_count
            or not np.isfinite(np.asarray(study_seconds, dtype=float)).all()
            or (np.asarray(study_seconds, dtype=float) <= 0).any()):
        raise ValueError('cache runtime receipt missing or incomplete')
    index = json.loads(index_path.read_text(encoding='utf-8'))
    if len(index) != study_count or len(set(index)) != study_count:
        raise ValueError('cache index count/UID uniqueness mismatch')
    if receipt.get('cache_index_sha256') and file_sha256(index_path) != receipt['cache_index_sha256']:
        raise ValueError('cache index hash mismatch')
    total_metadata = 0
    all_failures = []
    for shard in receipt.get('shards', []):
        shard_id, rows = int(shard['shard']), int(shard['rows'])
        metadata_path = root / f'metadata_{shard_id:03d}.jsonl'
        failures_path = root / f'failures_{shard_id:03d}.jsonl'
        if not metadata_path.is_file() or not failures_path.is_file():
            raise ValueError(f'missing metadata/failures for shard {shard_id}')
        failures = [json.loads(line) for line in failures_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        all_failures.extend(failures)
        records = [json.loads(line) for line in metadata_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(records) != rows:
            raise ValueError(f'metadata row mismatch for shard {shard_id}')
        if [f for record in records for f in record.get('failures', [])] != failures:
            raise ValueError('failure log differs from study metadata')
        by_uid = {str(record['StudyInstanceUID']): record for record in records}
        for failure in failures:
            record = by_uid.get(str(failure.get('StudyInstanceUID', '')))
            series = failure.get('SeriesInstanceUID')
            error = str(failure.get('error', ''))
            geometry_error = error == 'ValueError: declared plane differs from slice normal by over 30 degrees'
            decode_error = error.startswith('RuntimeError: decode failed at ') and error.endswith('.dcm: ValueError')
            if record is None or not series or not (geometry_error or decode_error):
                raise ValueError('unreviewed cache failure category')
            windows = [item for item in record.get('metadata', []) if item.get('slot') == failure.get('slot')]
            if not windows or any(not item.get('missing', False) and item.get('SeriesInstanceUID') == series for item in windows):
                raise ValueError('rejected series reused or missing slot provenance')
        for record in records:
            uid = str(record.get('StudyInstanceUID', ''))
            location = index.get(uid, {})
            if (location.get('shard') != shard_id
                    or location.get('row') != int(record['row'])
                    or location.get('laterality', 'unknown') not in {'L', 'R', 'unknown'}):
                raise ValueError(f'cache index/metadata mismatch for {uid}')
            if int(record.get('window_count', 0)) <= 0:
                raise ValueError(f'cache study has no valid windows: {uid}')
        total_metadata += len(records)
        image_path = root / f'images_{shard_id:03d}.npy'
        mask_path = root / f'mask_{shard_id:03d}.npy'
        if require_arrays and (not image_path.is_file() or not mask_path.is_file()):
            raise ValueError(f'image/mask shard missing for {shard_id}')
        if image_path.is_file() and mask_path.is_file():
            images = np.load(image_path, mmap_mode='r')
            masks = np.load(mask_path, mmap_mode='r')
            if images.shape != (rows, 24, 3, 336, 336) or images.dtype != np.uint8:
                raise ValueError(f'image shard shape/dtype mismatch for {shard_id}: {images.shape} {images.dtype}')
            if masks.shape != (rows, 24) or masks.dtype != np.float32:
                raise ValueError(f'mask shard shape/dtype mismatch for {shard_id}: {masks.shape} {masks.dtype}')
            if not np.isfinite(masks).all() or not np.isin(masks, [0, 1]).all() or not masks.any(axis=1).all():
                raise ValueError(f'invalid masks for shard {shard_id}')
            for record in records:
                windows = record.get('metadata')
                if windows is not None:
                    expected_mask = np.array([not item.get('missing', False) for item in windows])
                    if len(windows) != 24 or not np.array_equal(masks[int(record['row'])], expected_mask):
                        raise ValueError('window mask differs from provenance')
            if shard.get('images_sha256') and file_sha256(image_path) != shard['images_sha256']:
                raise ValueError(f'image shard hash mismatch for {shard_id}')
            if shard.get('mask_sha256') and file_sha256(mask_path) != shard['mask_sha256']:
                raise ValueError(f'mask shard hash mismatch for {shard_id}')
    if total_metadata != study_count:
        raise ValueError('metadata total does not match study count')
    affected = len({failure['StudyInstanceUID'] for failure in all_failures})
    if len(all_failures) != receipt.get('failure_count') or affected > .05 * study_count:
        raise ValueError('failure count mismatch or systematic anomaly threshold exceeded')
    montage = root / 'montage_000.png'
    montage_manifest = root / 'montage_manifest.json'
    if not montage.is_file() or not montage_manifest.is_file():
        raise ValueError('orientation montage/manifest missing')
    if receipt.get('montage_sha256') and file_sha256(montage) != receipt['montage_sha256']:
        raise ValueError('montage hash mismatch')
    if receipt.get('montage_manifest_sha256') and file_sha256(montage_manifest) != receipt['montage_manifest_sha256']:
        raise ValueError('montage manifest hash mismatch')
    return {
        'status': 'passed', 'study_count': study_count, 'metadata_rows': total_metadata,
        'rejected_series_count': len(all_failures), 'affected_studies': affected,
        'shards': len(receipt.get('shards', [])), 'require_arrays': bool(require_arrays),
        'receipt_status': receipt.get('status'), 'formal_training_ready': bool(receipt.get('formal_training_ready')),
    }


def validate_cache_collection(roots, expected_uids):
    """Verify all five disjoint private outputs against the frozen cohort."""
    roots = [Path(root) for root in roots]
    expected = sorted(map(str, expected_uids))
    if len(roots) != 5 or len(expected) != 4345 or len(set(expected)) != 4345:
        raise ValueError('expected five parts and 4345 distinct research studies')
    cohort_hash = hashlib.sha256('\n'.join(expected).encode()).hexdigest()
    parts, seen, config, digests = set(), set(), None, []
    for root in roots:
        validate_cache(root, require_arrays=True)
        receipt = json.loads((root/'cache_receipt.json').read_text())
        part = receipt.get('part_index')
        if receipt.get('part_count') != 5 or part not in range(5) or part in parts:
            raise ValueError('duplicate or incompatible cache partition')
        if receipt.get('cohort_uid_sha256') != cohort_hash:
            raise ValueError('cache cohort hash differs from frozen contract')
        if receipt['config'].get('geometry_version') != 'lps_plane_resample_v2':
            raise ValueError('obsolete geometry cache')
        if config is not None and receipt['config'] != config:
            raise ValueError('cache parts differ in preprocessing')
        config = receipt['config']
        index = json.loads((root/'cache_index.json').read_text())
        if set(index) != set(expected[part::5]) or seen & set(index):
            raise ValueError('partition UID coverage differs from frozen cohort')
        for name, key in [('cache_index.json','cache_index_sha256'),
                          ('metadata_000.jsonl','metadata_sha256')]:
            if receipt.get(key) != file_sha256(root/name):
                raise ValueError(f'cache artifact hash mismatch: {name}')
        for shard in receipt['shards']:
            if not shard.get('images_sha256') or not shard.get('mask_sha256'):
                raise ValueError('cache shard hashes missing')
        parts.add(part); seen.update(index)
        digests.append((part, file_sha256(root/'cache_receipt.json')))
    if seen != set(expected):
        raise ValueError('incomplete research cache')
    digest = hashlib.sha256(json.dumps(sorted(digests)).encode()).hexdigest()
    return {'status': 'passed', 'study_count': len(seen), 'part_count': 5,
            'cache_sha256': digest, 'geometry_version': 'lps_plane_resample_v2'}
