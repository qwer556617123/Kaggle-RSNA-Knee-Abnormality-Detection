"""Plane-preserving LPS resampling, with a shared physical centre."""
import numpy as np
from scipy.ndimage import map_coordinates
from .research_geometry import PLANE_TARGETS, _project_axis


def resample_series(images, geometry, plane, *, size=336, fov_mm=160.0):
    if len(images) != len(geometry.positions):
        raise ValueError('geometry/image count mismatch')
    horizontal, vertical, normal = PLANE_TARGETS[plane]
    if abs(geometry.normal @ normal) < np.cos(np.deg2rad(30)):
        raise ValueError('declared plane differs from slice normal by over 30 degrees')
    horizontal = _project_axis(horizontal, geometry.normal)
    vertical = _project_axis(vertical, geometry.normal)
    vertical -= (vertical @ horizontal) * horizontal
    vertical /= np.linalg.norm(vertical)
    rows, cols = images[0].shape
    centres = np.stack([geometry.pixel_to_patient(i, (rows-1)/2, (cols-1)/2)
                        for i in range(len(images))])
    centre = np.median(centres, axis=0)
    axis = (np.arange(size) + .5 - size/2) * fov_mm/size
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    offsets = yy[..., None]*vertical + xx[..., None]*horizontal
    output, transforms = [], []
    for i, image in enumerate(images):
        origin = geometry.positions[i]
        local_centre = centre + ((origin-centre) @ geometry.normal)*geometry.normal
        displacement = local_centre + offsets - origin
        source_rows = displacement @ geometry.column_direction / geometry.pixel_spacing[0]
        source_cols = displacement @ geometry.row_direction / geometry.pixel_spacing[1]
        values = map_coordinates(image, [source_rows, source_cols], order=1,
                                 mode='constant', cval=0, prefilter=False)
        output.append(np.clip(np.rint(values), 0, 255).astype(np.uint8))
        transforms.append({
            'sorted_slice': i, 'source_index': int(geometry.order[i]),
            'ipp': origin.tolist(), 'output_centre_lps': local_centre.tolist(),
            'output_origin_lps': (local_centre+offsets[0,0]).tolist(),
            'output_row_step_lps': (vertical*fov_mm/size).tolist(),
            'output_column_step_lps': (horizontal*fov_mm/size).tolist(),
            'source_valid_fraction': float(((source_rows >= 0) & (source_rows <= rows-1)
                                            & (source_cols >= 0) & (source_cols <= cols-1)).mean()),
        })
    return np.stack(output), {
        'version': 'lps_plane_resample_v2', 'target_horizontal': horizontal.tolist(),
        'target_vertical': vertical.tolist(), 'fov_mm': fov_mm,
        'source_iop': np.r_[geometry.row_direction, geometry.column_direction].tolist(),
        'source_pixel_spacing': geometry.pixel_spacing.tolist(), 'slices': transforms,
    }
