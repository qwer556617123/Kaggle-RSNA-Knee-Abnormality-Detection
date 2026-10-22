from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import SLOT_NAMES


@dataclass(frozen=True)
class SeriesChoice:
    study_uid: str
    series_uid: str
    slot: str
    path: Path
    file_count: int


def slot_for_series(plane: str, fluid_sensitive: Any) -> str | None:
    plane = str(plane).strip().capitalize()
    fluid = int(float(fluid_sensitive)) == 1
    lookup = {
        ("Axial", True): "Axial_Fluid",
        ("Sagittal", True): "Sagittal_Fluid",
        ("Coronal", True): "Coronal_Fluid",
        ("Sagittal", False): "Sagittal_NonFluid",
        ("Coronal", False): "Coronal_NonFluid",
    }
    return lookup.get((plane, fluid))


def choose_series(
    series_table: pd.DataFrame,
    dicom_root: str | Path,
    *,
    slots: Sequence[str] = SLOT_NAMES,
) -> dict[str, dict[str, SeriesChoice]]:
    required = {"StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"}
    missing = required.difference(series_table.columns)
    if missing:
        raise ValueError(f"series table missing columns: {sorted(missing)}")
    root = Path(dicom_root)
    output: dict[str, dict[str, SeriesChoice]] = {}
    for row in series_table.itertuples(index=False):
        study_uid = str(row.StudyInstanceUID)
        series_uid = str(row.SeriesInstanceUID)
        slot = slot_for_series(row.Anatomical_Plane, row.Fluid_Sensitive)
        if slot not in slots:
            continue
        path = root / study_uid / series_uid
        count = sum(1 for _ in path.glob("*.dcm")) if path.exists() else 0
        candidate = SeriesChoice(study_uid, series_uid, slot, path, count)
        previous = output.setdefault(study_uid, {}).get(slot)
        if previous is None or (candidate.file_count, candidate.series_uid) > (
            previous.file_count,
            previous.series_uid,
        ):
            output[study_uid][slot] = candidate
    return output


def _float_sequence(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray([float(item) for item in value], dtype=np.float64)
    except Exception:
        return None
    return array if np.isfinite(array).all() else None


def slice_coordinate(dataset: Any) -> float | None:
    position = _float_sequence(getattr(dataset, "ImagePositionPatient", None))
    orientation = _float_sequence(getattr(dataset, "ImageOrientationPatient", None))
    if position is not None and len(position) >= 3 and orientation is not None and len(orientation) >= 6:
        normal = np.cross(orientation[:3], orientation[3:6])
        norm = np.linalg.norm(normal)
        if norm > 1e-8:
            return float(np.dot(position[:3], normal / norm))
    location = getattr(dataset, "SliceLocation", None)
    try:
        return float(location)
    except (TypeError, ValueError):
        return None


def sort_dicom_files(files: Iterable[str | Path]) -> list[Path]:
    import pydicom

    records = []
    for path_like in files:
        path = Path(path_like)
        dataset = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        coordinate = slice_coordinate(dataset)
        try:
            instance = int(getattr(dataset, "InstanceNumber", 0))
        except (TypeError, ValueError):
            instance = 0
        records.append((path, coordinate, instance))
    if records and sum(record[1] is not None for record in records) >= max(2, len(records) // 2):
        return [record[0] for record in sorted(records, key=lambda item: (math.inf if item[1] is None else item[1], item[2], item[0].name))]
    return [record[0] for record in sorted(records, key=lambda item: (item[2], item[0].name))]


def evenly_spaced_indices(length: int, count: int) -> np.ndarray:
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    return np.rint(np.linspace(0, length - 1, count)).astype(np.int64)


def sample_slice_indices(length: int, count: int, mode: str = "even") -> np.ndarray:
    """Select ordered slice indices under an explicit, reproducible policy."""
    if mode == "even":
        return evenly_spaced_indices(length, count)
    if mode == "consecutive_midband":
        if length <= 0 or count <= 0:
            raise ValueError("length and count must be positive")
        if length < count:
            return evenly_spaced_indices(length, count)
        start = (length - count) // 2
        return np.arange(start, start + count, dtype=np.int64)
    raise ValueError(f"unsupported slice sampling mode {mode!r}")


def decode_pixels(dataset: Any) -> np.ndarray:
    try:
        raw_array = dataset.pixel_array
    except Exception as original_error:
        # Kaggle's pydicom build may not include pylibjpeg/GDCM. OpenCV can decode many
        # single-frame JPEG/JPEG2000 payloads directly from the encapsulated frame.
        try:
            import cv2
            from pydicom.encaps import generate_frames

            frame = next(generate_frames(dataset.PixelData, number_of_frames=1))
            raw_array = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if raw_array is None:
                raise ValueError("OpenCV returned no decoded pixels")
        except Exception:
            raise original_error
    array = np.asarray(raw_array, dtype=np.float32)
    if array.ndim != 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D DICOM slice, got {array.shape}")
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    array = array * slope + intercept
    if str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        array = np.nanmax(array) + np.nanmin(array) - array
    return array


def robust_uint8(volume: Sequence[np.ndarray], low: float = 1.0, high: float = 99.0) -> list[np.ndarray]:
    finite = np.concatenate([array[np.isfinite(array)].ravel()[::16] for array in volume if np.isfinite(array).any()])
    if finite.size == 0:
        return [np.zeros_like(array, dtype=np.uint8) for array in volume]
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        hi = lo + 1.0
    return [np.clip((array - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8) for array in volume]


def _pixel_spacing(dataset: Any) -> tuple[float, float] | None:
    spacing = _float_sequence(getattr(dataset, "PixelSpacing", None))
    if spacing is None or len(spacing) < 2 or (spacing[:2] <= 0).any():
        return None
    return float(spacing[0]), float(spacing[1])


def physical_center_crop(
    image: np.ndarray,
    spacing: tuple[float, float] | None,
    target_fov_mm: float = 170.0,
) -> np.ndarray:
    height, width = image.shape
    if spacing is None:
        side_h, side_w = height, width
    else:
        side_h = min(height, max(16, int(round(target_fov_mm / spacing[0]))))
        side_w = min(width, max(16, int(round(target_fov_mm / spacing[1]))))
    top = max(0, (height - side_h) // 2)
    left = max(0, (width - side_w) // 2)
    return image[top : top + side_h, left : left + side_w]


def resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((image_size, image_size), Image.Resampling.BILINEAR))


def volume_minmax_minus_one_to_one(volume: np.ndarray) -> np.ndarray:
    """Apply OrthoDiffusion's per-volume min-max normalization safely."""
    array = np.asarray(volume, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    low = float(finite.min())
    high = float(finite.max())
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=low, posinf=high, neginf=low)
    return ((array - low) * (2.0 / (high - low)) - 1.0).astype(np.float32, copy=False)


def load_series_volume(
    series_path: str | Path,
    *,
    slices: int = 16,
    image_size: int = 256,
    slice_sampling: str = "consecutive_midband",
) -> np.ndarray:
    """Decode one series using the official OrthoDiffusion volume contract."""
    import pydicom

    files = sort_dicom_files(Path(series_path).glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"no DICOM slices found under {series_path}")
    selected = [files[index] for index in sample_slice_indices(len(files), slices, slice_sampling)]
    planes: list[np.ndarray | None] = []
    for path in selected:
        try:
            planes.append(decode_pixels(pydicom.dcmread(path, force=True)))
        except Exception:
            planes.append(None)
    valid_indices = [index for index, plane in enumerate(planes) if plane is not None]
    if not valid_indices:
        raise RuntimeError(f"none of the sampled DICOM slices could be decoded under {series_path}")

    # A single corrupt slice should not create an artificial black plane. Reuse the
    # closest decoded anatomical neighbor, while still failing if the whole series is bad.
    recovered = []
    for index, plane in enumerate(planes):
        if plane is None:
            nearest = min(valid_indices, key=lambda candidate: (abs(candidate - index), candidate))
            plane = planes[nearest]
        assert plane is not None
        try:
            import cv2

            resized = cv2.resize(plane, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from PIL import Image

            resized = np.asarray(
                Image.fromarray(plane).resize((image_size, image_size), Image.Resampling.BILINEAR)
            )
        recovered.append(np.asarray(resized, dtype=np.float32))
    volume = volume_minmax_minus_one_to_one(np.stack(recovered, axis=0))
    return volume[None, ...]


def load_series_groups(
    series_path: str | Path,
    *,
    slices: int = 15,
    group_size: int = 3,
    image_size: int = 196,
    target_fov_mm: float = 170.0,
    slice_sampling: str = "even",
) -> np.ndarray:
    import pydicom

    if slices % group_size:
        raise ValueError("slices must be divisible by group_size")
    files = sort_dicom_files(Path(series_path).glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"no DICOM slices found under {series_path}")
    selected = [files[index] for index in sample_slice_indices(len(files), slices, slice_sampling)]
    datasets = []
    planes: list[np.ndarray | None] = []
    for path in selected:
        try:
            dataset = pydicom.dcmread(path, force=True)
            plane = decode_pixels(dataset)
        except Exception:
            dataset, plane = None, None
        datasets.append(dataset)
        planes.append(plane)
    valid = [plane for plane in planes if plane is not None]
    if not valid:
        raise RuntimeError(f"none of the sampled DICOM slices could be decoded under {series_path}")
    reference_shape = valid[0].shape
    volume = [
        plane if plane is not None and plane.shape == reference_shape else np.zeros(reference_shape, dtype=np.float32)
        for plane in planes
    ]
    decoded = robust_uint8(volume)
    resized = [
        resize_image(
            physical_center_crop(image, _pixel_spacing(dataset) if dataset is not None else None, target_fov_mm),
            image_size,
        )
        for image, dataset in zip(decoded, datasets)
    ]
    array = np.stack(resized).reshape(slices // group_size, group_size, image_size, image_size)
    return array.astype(np.uint8, copy=False)


def build_study_tensor(
    study_uid: str,
    choices: Mapping[str, Mapping[str, SeriesChoice]],
    *,
    slots: Sequence[str] = SLOT_NAMES,
    slices: int = 15,
    group_size: int = 3,
    image_size: int = 196,
    target_fov_mm: float | Sequence[float] = 170.0,
    slice_sampling: str = "even",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    groups = slices // group_size
    if isinstance(target_fov_mm, (int, float)):
        fov_by_slot = [float(target_fov_mm)] * len(slots)
    else:
        fov_by_slot = [float(value) for value in target_fov_mm]
        if len(fov_by_slot) != len(slots):
            raise ValueError("target_fov_mm sequence must align with slots")
    if any(not math.isfinite(value) or value <= 0 for value in fov_by_slot):
        raise ValueError("target_fov_mm values must be finite and positive")
    tensor = np.zeros((len(slots), groups, group_size, image_size, image_size), dtype=np.uint8)
    mask = np.zeros(len(slots), dtype=np.float32)
    failures = []
    available = choices.get(str(study_uid), {})
    for slot_index, slot in enumerate(slots):
        choice = available.get(slot)
        if choice is None:
            continue
        try:
            tensor[slot_index] = load_series_groups(
                choice.path,
                slices=slices,
                group_size=group_size,
                image_size=image_size,
                target_fov_mm=fov_by_slot[slot_index],
                slice_sampling=slice_sampling,
            )
            mask[slot_index] = 1.0
        except Exception as exc:
            failures.append(
                {
                    "StudyInstanceUID": str(study_uid),
                    "SeriesInstanceUID": choice.series_uid,
                    "slot": slot,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return tensor, mask, failures
