import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rsna_knee.dicom import (
    SeriesChoice,
    build_study_tensor,
    evenly_spaced_indices,
    physical_center_crop,
    sample_slice_indices,
    slice_coordinate,
    slot_for_series,
    volume_minmax_minus_one_to_one,
)


class FakeDataset:
    ImagePositionPatient = [0, 0, 12]
    ImageOrientationPatient = [1, 0, 0, 0, 1, 0]


class DicomTests(unittest.TestCase):
    def test_physical_coordinate(self):
        self.assertAlmostEqual(slice_coordinate(FakeDataset()), 12.0)

    def test_single_slice_repeats_safely(self):
        np.testing.assert_array_equal(evenly_spaced_indices(1, 15), np.zeros(15, dtype=int))

    def test_consecutive_midband_sampling(self):
        np.testing.assert_array_equal(
            sample_slice_indices(21, 9, "consecutive_midband"),
            np.arange(6, 15),
        )
        np.testing.assert_array_equal(
            sample_slice_indices(3, 5, "consecutive_midband"),
            np.array([0, 0, 1, 2, 2]),
        )

    def test_rejects_unknown_sampling_policy(self):
        with self.assertRaises(ValueError):
            sample_slice_indices(10, 3, "filename")

    def test_slots(self):
        self.assertEqual(slot_for_series("axial", 1), "Axial_Fluid")
        self.assertEqual(slot_for_series("Sagittal", 0), "Sagittal_NonFluid")
        self.assertIsNone(slot_for_series("Axial", 0))

    def test_physical_crop(self):
        image = np.zeros((512, 384), dtype=np.uint8)
        cropped = physical_center_crop(image, (0.5, 0.5), target_fov_mm=170)
        self.assertEqual(cropped.shape, (340, 340))

    def test_volume_minmax_normalization_is_finite(self):
        volume = np.array([[[np.nan, 0.0], [5.0, np.inf]]], dtype=np.float32)
        normalized = volume_minmax_minus_one_to_one(volume)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertAlmostEqual(float(normalized.min()), -1.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)

    def test_constant_volume_normalizes_to_neutral(self):
        normalized = volume_minmax_minus_one_to_one(np.full((2, 3, 4), 7.0, dtype=np.float32))
        np.testing.assert_array_equal(normalized, np.zeros((2, 3, 4), dtype=np.float32))

    def test_multiscale_fov_aligns_with_duplicate_slots(self):
        choice = SeriesChoice("study", "series", "Axial_Fluid", Path("series"), 9)
        calls = []

        def fake_load(*args, **kwargs):
            calls.append(kwargs["target_fov_mm"])
            return np.zeros((1, 3, 8, 8), dtype=np.uint8)

        with patch("rsna_knee.dicom.load_series_groups", side_effect=fake_load):
            tensor, mask, failures = build_study_tensor(
                "study",
                {"study": {"Axial_Fluid": choice}},
                slots=("Axial_Fluid", "Axial_Fluid"),
                slices=3,
                group_size=3,
                image_size=8,
                target_fov_mm=(160.0, 112.0),
            )
        self.assertEqual(calls, [160.0, 112.0])
        self.assertEqual(tensor.shape, (2, 1, 3, 8, 8))
        np.testing.assert_array_equal(mask, np.ones(2, dtype=np.float32))
        self.assertEqual(failures, [])

    def test_multiscale_fov_length_must_match_slots(self):
        with self.assertRaises(ValueError):
            build_study_tensor(
                "study",
                {},
                slots=("Axial_Fluid", "Axial_Fluid"),
                target_fov_mm=(160.0,),
            )


if __name__ == "__main__":
    unittest.main()
