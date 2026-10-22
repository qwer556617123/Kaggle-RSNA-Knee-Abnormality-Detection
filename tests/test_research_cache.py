import unittest
from types import SimpleNamespace

import numpy as np

from rsna_knee.research_cache import ResearchCacheConfig, physical_triplets_sorted
from rsna_knee.research_geometry import apply_canonical_plane, canonical_plane_transform, validate_series_geometry


class ResearchCacheTests(unittest.TestCase):
    def headers(self, orientation=(1,0,0,0,1,0), positions=(0,1,2)):
        row, column = np.asarray(orientation[:3]), np.asarray(orientation[3:])
        normal = np.cross(row,column)
        return [SimpleNamespace(StudyInstanceUID='s',SeriesInstanceUID='ser',Rows=2,Columns=3,
                                PixelSpacing=[1,2],ImageOrientationPatient=list(orientation),
                                ImagePositionPatient=(normal*z).tolist()) for z in positions]

    def test_frozen_cache_budget(self):
        config = ResearchCacheConfig()
        self.assertEqual(sum(config.evaluation_counts),24)
        self.assertEqual(sum(config.training_counts),12)
        with self.assertRaises(ValueError):
            ResearchCacheConfig(evaluation_counts=(1,1,1,1,1))

    def test_axial_identity(self):
        transform = canonical_plane_transform(validate_series_geometry(self.headers()),'Axial')
        self.assertFalse(transform['transpose'])
        image = np.arange(6).reshape(2,3)
        np.testing.assert_array_equal(apply_canonical_plane(image,transform),image)

    def test_axis_flips_are_deterministic(self):
        headers = self.headers(orientation=(-1,0,0,0,-1,0))
        transform = canonical_plane_transform(validate_series_geometry(headers),'Axial')
        self.assertTrue(transform['flip_vertical'])
        self.assertTrue(transform['flip_horizontal'])
        image = np.arange(6).reshape(2,3)
        np.testing.assert_array_equal(apply_canonical_plane(image,transform),image[::-1,::-1])

    def test_oblique_grid_requires_explicit_resampling(self):
        angle = np.deg2rad(45)
        orientation = (np.cos(angle),np.sin(angle),0,-np.sin(angle),np.cos(angle),0)
        with self.assertRaises(ValueError):
            canonical_plane_transform(validate_series_geometry(self.headers(orientation=orientation)),'Axial')

    def test_wrong_plane_rejected(self):
        with self.assertRaises(ValueError):
            canonical_plane_transform(validate_series_geometry(self.headers()),'Sagittal')

    def test_triplet_mapping_keeps_sorted_array_and_source_indices_distinct(self):
        # Input files are deliberately out of physical order: source indices
        # remain traceable while the decoded array is indexed in sorted order.
        geometry = validate_series_geometry(self.headers(positions=(4,0,2,6,8)))
        source, sorted_indices = physical_triplets_sorted(geometry, 3)
        np.testing.assert_array_equal(source, [[1,2,0],[2,0,3],[0,3,4]])
        np.testing.assert_array_equal(sorted_indices, [[0,1,2],[1,2,3],[2,3,4]])


if __name__ == '__main__':
    unittest.main()
