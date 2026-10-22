import unittest
from types import SimpleNamespace
import numpy as np
from rsna_knee.research_geometry import validate_series_geometry
from rsna_knee.research_resample import resample_series


class ResamplingTests(unittest.TestCase):
    def test_shifted_slice_origins_align_a_physical_landmark(self):
        headers, images = [], []
        for i, shift in enumerate((-2, 0, 2)):
            headers.append(SimpleNamespace(
                StudyInstanceUID='s', SeriesInstanceUID='a', Rows=9, Columns=9,
                PixelSpacing=[1., 1.], ImageOrientationPatient=[1,0,0,0,1,0],
                ImagePositionPatient=[shift, 0, i]))
            image = np.zeros((9,9), np.float32)
            image[4, 4-shift] = 255
            images.append(image)
        result, transform = resample_series(images, validate_series_geometry(headers),
                                            'Axial', size=9, fov_mm=9)
        for image in result:
            self.assertEqual(np.unravel_index(image.argmax(), image.shape), (4,4))
        self.assertEqual(transform['version'], 'lps_plane_resample_v2')

    def test_oblique_in_plane_rotation_uses_target_axes(self):
        v = 2**-.5
        headers = [SimpleNamespace(
            StudyInstanceUID='s', SeriesInstanceUID='a', Rows=21, Columns=21,
            PixelSpacing=[1., 1.], ImageOrientationPatient=[v,v,0,-v,v,0],
            ImagePositionPatient=[0,0,i]) for i in range(3)]
        result, transform = resample_series([np.ones((21,21))*100]*3,
                                            validate_series_geometry(headers),
                                            'Axial', size=9, fov_mm=9)
        np.testing.assert_allclose(transform['target_horizontal'], [1,0,0])
        np.testing.assert_allclose(transform['target_vertical'], [0,1,0])
        self.assertTrue(np.isfinite(result).all())
