"""Strict physical geometry for future local-evidence caches.

Separate from legacy inference: missing geometry must not silently become an
anatomical ROI. This validates coordinates, not full canonical resampling.
"""
from dataclasses import dataclass

import numpy as np


PLANE_TARGETS = {
    'Axial': (np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])),
    'Coronal': (np.array([1., 0., 0.]), np.array([0., 0., -1.]), np.array([0., 1., 0.])),
    'Sagittal': (np.array([0., 1., 0.]), np.array([0., 0., -1.]), np.array([1., 0., 0.])),
}


@dataclass(frozen=True)
class SeriesGeometry:
    order: np.ndarray
    positions: np.ndarray
    coordinates: np.ndarray
    row_direction: np.ndarray
    column_direction: np.ndarray
    normal: np.ndarray
    pixel_spacing: np.ndarray
    study_uid: str
    series_uid: str

    def pixel_to_patient(self, sorted_slice, row, column):
        # IOP row direction follows increasing array COLUMN (DICOM C.7.6.2).
        return (self.positions[sorted_slice] + row*self.pixel_spacing[0]*self.column_direction
                + column*self.pixel_spacing[1]*self.row_direction)


def _vector(header, name, length):
    try:
        value = np.asarray(getattr(header,name),dtype=float)
    except (AttributeError,TypeError,ValueError) as error:
        raise ValueError(f'missing/invalid {name}') from error
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f'missing/invalid {name}')
    return value


def validate_series_geometry(headers):
    """Return one series sorted in patient LPS coordinates, never by filename."""
    if not headers:
        raise ValueError('empty series')
    first = headers[0]
    study, series = str(getattr(first,'StudyInstanceUID','')), str(getattr(first,'SeriesInstanceUID',''))
    if not study or not series:
        raise ValueError('missing study/series identity')
    orientation = _vector(first,'ImageOrientationPatient',6)
    # DICOM Image Orientation Patient stores row direction followed by column
    # direction. PixelSpacing is row spacing followed by column spacing.
    row, column = orientation[:3],orientation[3:]
    if not np.allclose([np.linalg.norm(column),np.linalg.norm(row),column@row],[1,1,0],atol=1e-4,rtol=0):
        raise ValueError('nonorthonormal DICOM orientation')
    normal = np.cross(row,column)
    normal /= np.linalg.norm(normal)
    spacing = _vector(first,'PixelSpacing',2)
    if (spacing <= 0).any():
        raise ValueError('nonpositive pixel spacing')
    positions = []
    dimensions = (getattr(first,'Rows',0),getattr(first,'Columns',0))
    if any(int(x) <= 0 for x in dimensions):
        raise ValueError('invalid slice dimensions')
    for header in headers:
        if str(getattr(header,'StudyInstanceUID','')) != study or str(getattr(header,'SeriesInstanceUID','')) != series:
            raise ValueError('cross-study/series geometry')
        if (getattr(header,'Rows',0),getattr(header,'Columns',0)) != dimensions:
            raise ValueError('inconsistent slice dimensions')
        if not np.allclose(_vector(header,'ImageOrientationPatient',6),orientation,atol=1e-4,rtol=0):
            raise ValueError('varying slice orientation requires explicit resampling')
        if not np.allclose(_vector(header,'PixelSpacing',2),spacing,atol=1e-5,rtol=0):
            raise ValueError('varying pixel spacing requires explicit resampling')
        positions.append(_vector(header,'ImagePositionPatient',3))
    positions = np.asarray(positions)
    projected = positions @ normal
    order = np.argsort(projected,kind='stable')
    coordinates = projected[order]
    if len(coordinates)>1 and (np.diff(coordinates) <= 1e-4).any():
        raise ValueError('duplicate slice planes require explicit acquisition selection')
    return SeriesGeometry(order,positions[order],coordinates,row,column,normal,spacing,study,series)


def physical_triplets(geometry, count, *, max_gap_ratio=2.5):
    """Select real neighboring slices within a single series, by physical span.

Returns original input indices, not indices into a flattened multi-series bag.
Short or gapped series fail for explicit caller handling; no fabricated RGB.
"""
    if not isinstance(count,int) or count < 1 or max_gap_ratio < 1 or not np.isfinite(max_gap_ratio):
        raise ValueError('invalid physical sampling configuration')
    positions = geometry.coordinates
    if len(positions) < 3:
        raise ValueError('fewer than three slices cannot form a physical triplet')
    gaps = np.diff(positions)
    # The smaller half is robust when a short series contains a large missing
    # interval; an ordinary median of [1,100] would incorrectly permit the gap.
    baseline = np.median(np.sort(gaps)[:max(1,len(gaps)//2)])
    threshold = float(baseline)*max_gap_ratio
    if np.any(gaps > threshold):
        raise ValueError('abnormal inter-plane gap')
    candidates = np.array([i for i in range(1,len(positions)-1)
                           if gaps[i-1] <= threshold and gaps[i] <= threshold],dtype=int)
    if not len(candidates):
        raise ValueError('no contiguous physical triplets')
    targets = np.linspace(positions[candidates[0]],positions[candidates[-1]],count)
    centers = candidates[np.abs(positions[candidates,None]-targets).argmin(axis=0)]
    sorted_indices = centers[:,None] + np.array([-1,0,1])[None]
    return geometry.order[sorted_indices]


def _project_axis(axis, normal):
    projected = np.asarray(axis, dtype=float) - np.dot(axis, normal) * normal
    norm = np.linalg.norm(projected)
    if norm < 1e-8:
        raise ValueError('target axis is normal to source plane')
    return projected / norm


def canonical_plane_transform(geometry, plane, *, max_angle_degrees=30.0):
    """Describe a deterministic in-plane transpose/flip to patient axes.

    This intentionally supports only pixel-grid-preserving transforms. Oblique
    in-plane resampling is rejected until an explicit interpolating transform
    is added; the cache must never silently rotate an anatomy axis.
    """
    try:
        target_h, target_v, expected_normal = PLANE_TARGETS[str(plane).capitalize()]
    except KeyError as error:
        raise ValueError(f'unsupported anatomical plane {plane!r}') from error
    source_normal = geometry.normal
    if abs(float(np.dot(source_normal, expected_normal))) < np.cos(np.deg2rad(max_angle_degrees)):
        raise ValueError('series normal is inconsistent with declared plane')
    target_h = _project_axis(target_h, source_normal)
    target_v = _project_axis(target_v, source_normal)
    target_v = target_v - np.dot(target_v, target_h) * target_h
    target_v /= np.linalg.norm(target_v)
    source_axes = (geometry.column_direction, geometry.row_direction)  # array axis 0, 1
    scores = np.abs(np.asarray(source_axes) @ np.stack([target_v, target_h]).T)
    # Each target output axis must map to a different source image axis.
    source_for_v = int(np.argmax(scores[:, 0]))
    source_for_h = int(np.argmax(scores[:, 1]))
    if source_for_v == source_for_h or scores[source_for_v, 0] < np.cos(np.deg2rad(max_angle_degrees)) or scores[source_for_h, 1] < np.cos(np.deg2rad(max_angle_degrees)):
        raise ValueError('in-plane direction requires explicit resampling')
    if {source_for_v, source_for_h} != {0, 1}:
        raise ValueError('invalid source-axis mapping')
    return {
        'plane': str(plane).capitalize(), 'transpose': bool((source_for_v, source_for_h) == (1, 0)),
        'flip_vertical': bool(np.dot(source_axes[source_for_v], target_v) < 0),
        'flip_horizontal': bool(np.dot(source_axes[source_for_h], target_h) < 0),
        'normal_cosine': float(abs(np.dot(source_normal, expected_normal))),
        'target_horizontal': target_h.tolist(), 'target_vertical': target_v.tolist(),
    }


def apply_canonical_plane(image, transform):
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError('canonical plane expects a 2D image')
    if transform['transpose']:
        array = array.T
    if transform['flip_vertical']:
        array = array[::-1, :]
    if transform['flip_horizontal']:
        array = array[:, ::-1]
    return np.ascontiguousarray(array)
