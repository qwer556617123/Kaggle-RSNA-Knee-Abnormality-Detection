import unittest

import numpy as np

from rsna_knee.constants import SLOT_NAMES
from rsna_knee.research_roi import apply_roi_crops, crop_offset, local_window_mask, validate_window_metadata


def metadata():
    result = []
    for slot, count in zip(SLOT_NAMES, (4, 6, 6, 4, 4)):
        result.extend([{
            'slot': slot,
            'source_detail': {
                'pixel_spacing': [0.5, 0.5],
                'transform': {'fov_mm': 160.0, 'target_horizontal': 'L-R', 'target_vertical': 'P-A'},
            },
        }] * count)
    return result


class ResearchRoiTests(unittest.TestCase):
    def test_metadata_requires_order_and_canonical_transform(self):
        valid = metadata()
        validate_window_metadata(valid)
        invalid = metadata()
        invalid[3] = dict(invalid[3], slot='wrong')
        with self.assertRaisesRegex(ValueError, 'slot mismatch'):
            validate_window_metadata(invalid)

    def test_equal_compute_crop_pattern_and_no_flip(self):
        indices = np.arange(24)
        local = local_window_mask(indices)
        self.assertEqual(int(local.sum()), 12)
        image = np.broadcast_to(np.arange(336, dtype=np.float32)[None, None, None, :], (24, 3, 336, 336)).copy()
        center = apply_roi_crops(image, indices, 'center')
        anatomic = apply_roi_crops(image, indices, 'anatomic')
        self.assertEqual(center.shape, image.shape)
        self.assertEqual(anatomic.shape, image.shape)
        self.assertTrue(np.array_equal(center[~local], image[~local]))
        self.assertTrue(np.array_equal(anatomic[~local], image[~local]))
        self.assertFalse(np.array_equal(center[local], anatomic[local]))
        self.assertGreater(float(anatomic[1, 0, 0, -1]), float(anatomic[1, 0, 0, 0]))

    def test_registered_offsets_are_not_target_or_report_dependent(self):
        self.assertEqual(crop_offset(0, 1, 'center'), (0.0, 0.0))
        self.assertNotEqual(crop_offset(0, 1, 'anatomic'), crop_offset(0, 3, 'anatomic'))
        with self.assertRaisesRegex(ValueError, 'invalid protocol'):
            crop_offset(5, 0, 'anatomic')


if __name__ == '__main__':
    unittest.main()
