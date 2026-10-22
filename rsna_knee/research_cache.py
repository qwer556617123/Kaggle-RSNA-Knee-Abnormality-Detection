"""Traceable raw-DICOM cache for the clean research arms."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import SLOT_NAMES
from .dicom import decode_pixels, robust_uint8, resize_image, _pixel_spacing
from .research_geometry import apply_canonical_plane, canonical_plane_transform, physical_triplets, validate_series_geometry
from .research_resample import resample_series


@dataclass(frozen=True)
class ResearchCacheConfig:
    image_size: int = 336
    target_fov_mm: float = 160.0
    evaluation_counts: tuple[int, ...] = (4, 6, 6, 4, 4)
    training_counts: tuple[int, ...] = (2, 3, 3, 2, 2)
    slots: tuple[str, ...] = SLOT_NAMES
    max_shard_bytes: int = 7_500_000_000

    def __post_init__(self):
        if len(self.slots) != 5 or len(self.evaluation_counts) != len(self.slots) or len(self.training_counts) != len(self.slots):
            raise ValueError('slot/count configuration mismatch')
        if sum(self.evaluation_counts) != 24 or sum(self.training_counts) != 12:
            raise ValueError('unexpected window budget')
        if self.image_size != 336 or self.target_fov_mm != 160.0:
            raise ValueError('research cache resolution/FOV is frozen')


def _file_hash(paths):
    return hashlib.sha256('\n'.join(str(p) for p in paths).encode()).hexdigest()


def series_candidates(series_table, dicom_root, study_uid, slot):
    """Return all candidates, largest-first; caller may fall through failures."""
    root = Path(dicom_root)
    rows = series_table[(series_table['StudyInstanceUID'].astype(str) == str(study_uid)) & (series_table['slot'] == slot)]
    result = []
    for row in rows.itertuples(index=False):
        path = root / str(study_uid) / str(row.SeriesInstanceUID)
        files = sorted(path.glob('*.dcm')) if path.is_dir() else []
        result.append({'study_uid': str(study_uid), 'series_uid': str(row.SeriesInstanceUID), 'slot': slot,
                       'path': path, 'files': files, 'file_count': len(files)})
    return sorted(result, key=lambda item: (-item['file_count'], item['series_uid']))


def _common_crop(array, spacing, target_fov_mm):
    height, width = array.shape[-2:]
    sy, sx = spacing if spacing is not None else (target_fov_mm / height, target_fov_mm / width)
    crop_h = min(height, max(16, int(round(target_fov_mm / sy))))
    crop_w = min(width, max(16, int(round(target_fov_mm / sx))))
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    return array[..., top:top+crop_h, left:left+crop_w], (top, left, crop_h, crop_w)


def load_research_series(candidate, plane, config: ResearchCacheConfig, *, image_size=None, target_fov_mm=None):
    import pydicom
    files = candidate['files']
    if len(files) < 3:
        raise ValueError('series has fewer than three files')
    headers = [pydicom.dcmread(path, stop_before_pixels=True, force=True) for path in files]
    geometry = validate_series_geometry(headers)
    ordered_files = [files[index] for index in geometry.order]
    decoded, syntax_counts, lateralities = [], {}, set()
    for index, path in enumerate(ordered_files):
        dataset = pydicom.dcmread(path, force=True)
        try:
            pixels = decode_pixels(dataset)
            # Pixel padding is expressed in stored values; undo the rescale
            # for this mask, and restore MONOCHROME1 inversion when needed.
            padding = getattr(dataset, 'PixelPaddingValue', None)
            if padding is not None:
                raw = np.asarray(dataset.pixel_array)
                limit = getattr(dataset, 'PixelPaddingRangeLimit', padding)
                lo, hi = sorted((float(padding), float(limit)))
                pixels = pixels.copy()
                pixels[(raw >= lo) & (raw <= hi)] = np.nan
            if not np.isfinite(pixels).any():
                raise ValueError('slice has no valid pixels')
            decoded.append(pixels)
            syntax = str(dataset.file_meta.TransferSyntaxUID)
            syntax_counts[syntax] = syntax_counts.get(syntax, 0) + 1
            laterality = str(getattr(dataset, 'ImageLaterality', getattr(dataset, 'Laterality', ''))).upper()
            if laterality in ('L', 'R'):
                lateralities.add(laterality)
        except Exception as error:
            raise RuntimeError(f'decode failed at {path.name}: {type(error).__name__}') from error
    if len({plane.shape for plane in decoded}) != 1:
        raise ValueError('decoded plane dimensions differ')
    finite = np.concatenate([x[np.isfinite(x)] for x in decoded])
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        raise ValueError('series has no robust intensity contrast')
    normalized = [np.clip(np.nan_to_num((x-low)*255/(high-low), nan=0, posinf=255, neginf=0), 0, 255)
                  for x in decoded]
    output_size = config.image_size if image_size is None else int(image_size)
    output_fov = config.target_fov_mm if target_fov_mm is None else float(target_fov_mm)
    if output_size < 16 or output_fov <= 0:
        raise ValueError('invalid native resampling size or field of view')
    resized, transform = resample_series(normalized, geometry, plane,
                                         size=output_size, fov_mm=output_fov)
    # geometry.order maps sorted physical positions to the original file
    # list. The decoded/resized array above is already in sorted order, so
    # convert the public original-index triplets back to sorted indices before
    # indexing it. Keeping physical_triplets' original-index contract is
    # important for callers that need to trace source files.
    triplets_original, triplets = physical_triplets_sorted(
        geometry, max(config.evaluation_counts[SLOT_NAMES.index(candidate['slot'])], 1)
    )
    windows = resized[triplets]
    coordinates = geometry.coordinates[np.asarray(triplets)]
    return windows, coordinates, {'series_uid': candidate['series_uid'], 'files': [str(p) for p in ordered_files],
                                  'source_triplets': triplets_original.tolist(),
                                  'sorted_triplets': triplets.tolist(),
                                  'source_shape': list(decoded[0].shape),
                                  'resampled_shape': [output_size, output_size],
                                  'transform': transform, 'crop': None, 'pixel_spacing': geometry.pixel_spacing.tolist(),
                                  'transfer_syntax_counts': syntax_counts,
                                  'laterality': next(iter(lateralities)) if len(lateralities) == 1 else 'unknown',
                                  'intensity_percentiles': [float(low), float(high)],
                                  'geometry_hash': _file_hash(ordered_files)}


def physical_triplets_sorted(geometry, count):
    """Return traceable original-file triplets and sorted-array triplets."""
    triplets_original = np.asarray(physical_triplets(geometry, count), dtype=int)
    inverse_order = np.empty_like(geometry.order)
    inverse_order[np.asarray(geometry.order, dtype=int)] = np.arange(len(geometry.order))
    return triplets_original, inverse_order[triplets_original]


def build_study_cache(study_uid, series_table, dicom_root, config=ResearchCacheConfig()):
    chunks = [np.zeros((3, config.image_size, config.image_size), dtype=np.uint8) for _ in range(sum(config.evaluation_counts))]
    masks = np.zeros(sum(config.evaluation_counts), dtype=np.float32)
    metadata, failures = [], []
    offset = 0
    for slot_index, slot in enumerate(config.slots):
        candidates = series_candidates(series_table, dicom_root, study_uid, slot)
        loaded = None
        for candidate in candidates:
            try:
                loaded = load_research_series(candidate, slot.split('_')[0], config)
                break
            except Exception as error:
                failures.append({'StudyInstanceUID': str(study_uid), 'slot': slot,
                                 'SeriesInstanceUID': candidate['series_uid'], 'error': f'{type(error).__name__}: {error}'})
        count = config.evaluation_counts[slot_index]
        if loaded is None:
            metadata.extend([{'StudyInstanceUID': str(study_uid), 'slot': slot, 'SeriesInstanceUID': '', 'missing': True} for _ in range(count)])
            offset += count
            continue
        windows, coordinates, detail = loaded
        if len(windows) != count:
            raise RuntimeError('series window count violated fixed slot budget')
        for window_index, (window, coords) in enumerate(zip(windows, coordinates)):
            chunks[offset] = window
            masks[offset] = 1.0
            metadata.append({'StudyInstanceUID': str(study_uid), 'slot': slot,
                             'SeriesInstanceUID': detail['series_uid'], 'depth_start': float(coords[0]),
                             'depth_end': float(coords[-1]), 'window_index_in_series': window_index,
                             'source_triplet': detail['source_triplets'][window_index],
                             'sorted_triplet': detail['sorted_triplets'][window_index],
                             'source_detail': detail})
            offset += 1
        if len(metadata) < offset:
            metadata.extend([{'StudyInstanceUID': str(study_uid), 'slot': slot, 'SeriesInstanceUID': '', 'missing': True} for _ in range(offset-len(metadata))])
    if not np.any(masks):
        raise RuntimeError(f'all protocol slots failed for study {study_uid}')
    tensor = np.stack(chunks, axis=0)
    return tensor, masks, metadata, failures
