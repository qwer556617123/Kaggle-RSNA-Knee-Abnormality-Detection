import unittest
from types import SimpleNamespace

import numpy as np

from rsna_knee.research_geometry import validate_series_geometry, physical_triplets


class ResearchGeometryTests(unittest.TestCase):
    def headers(self, positions=(4,0,2,6,8), orientation=(1,0,0,0,1,0)):
        normal = np.cross(orientation[:3],orientation[3:])
        return [SimpleNamespace(StudyInstanceUID='study',SeriesInstanceUID='series',
                                Rows=10,Columns=12,PixelSpacing=[.5,2.],
                                ImageOrientationPatient=orientation,ImagePositionPatient=normal*z,
                                InstanceNumber=100-i) for i,z in enumerate(positions)]

    def test_physical_sort_ignores_instance_number(self):
        geometry = validate_series_geometry(self.headers())
        np.testing.assert_array_equal(geometry.order,[1,2,0,3,4])
        np.testing.assert_array_equal(physical_triplets(geometry,3),[[1,2,0],[2,0,3],[0,3,4]])

    def test_anisotropic_pixel_to_lps(self):
        geometry = validate_series_geometry(self.headers())
        np.testing.assert_allclose(geometry.pixel_to_patient(0,4,3),[6,2,0])

    def test_oblique_orientation_uses_normal_projection(self):
        v = 2**-.5
        orientation = [v,v,0,0,0,1]
        geometry = validate_series_geometry(self.headers(orientation=orientation))
        np.testing.assert_allclose(geometry.coordinates,[0,2,4,6,8])
        np.testing.assert_allclose(geometry.pixel_to_patient(0,2,1),[2*v,2*v,1])

    def test_mixed_series_and_missing_geometry_rejected(self):
        headers = self.headers()
        headers[-1].SeriesInstanceUID = 'different'
        with self.assertRaises(ValueError):
            validate_series_geometry(headers)
        headers = self.headers()
        del headers[0].ImagePositionPatient
        with self.assertRaises(ValueError):
            validate_series_geometry(headers)

    def test_duplicate_plane_rejected(self):
        with self.assertRaises(ValueError):
            validate_series_geometry(self.headers((0,1,1,2)))

    def test_short_series_is_not_replicated(self):
        geometry = validate_series_geometry(self.headers((0,1)))
        with self.assertRaises(ValueError):
            physical_triplets(geometry,3)

    def test_triplets_do_not_cross_large_gap(self):
        headers = self.headers((0,1,2,20,21,22))
        geometry = validate_series_geometry(headers)
        with self.assertRaises(ValueError):
            physical_triplets(geometry,10)

    def test_nonorthogonal_orientation_rejected(self):
        with self.assertRaises(ValueError):
            validate_series_geometry(self.headers(orientation=(1,0,0,1,1,0)))

    def test_short_series_large_gap_rejected(self):
        geometry = validate_series_geometry(self.headers((0,1,100)))
        with self.assertRaises(ValueError):
            physical_triplets(geometry,1)


if __name__ == '__main__':
    unittest.main()
