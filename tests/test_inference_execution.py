import tempfile
import unittest
from pathlib import Path
import numpy as np
from rsna_knee.inference_execution import competition_root, execution_splits, validate_probabilities


class InferenceExecutionTests(unittest.TestCase):
    def test_hidden_test_mount_needs_no_training_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'competitions/rsna-knee-abnormality-detection'; root.mkdir(parents=True)
            (root / 'test.csv').write_text('StudyInstanceUID\nu1\n')
            (root / 'test_series.csv').write_text('StudyInstanceUID,SeriesInstanceUID\nu1,s1\n')
            self.assertEqual(competition_root(folder), root)
            self.assertEqual(execution_splits(True), ('test',))
            self.assertEqual(execution_splits(False), ('train', 'test'))

    def test_one_or_two_hidden_studies_valid_but_preview_cannot_skip_gate(self):
        for count in (1, 2):
            values = np.full((count, 12), .4)
            validate_probabilities(values, public_preview=False)
            with self.assertRaises(ValueError):
                validate_probabilities(values, public_preview=True)
        for value in (np.nan, np.inf, -.1, 1.1):
            with self.assertRaises(ValueError):
                validate_probabilities(np.full((1, 12), value), public_preview=False)
        with self.assertRaises(ValueError):
            validate_probabilities(np.empty((0, 12)), public_preview=False)
        with self.assertRaises(ValueError):
            validate_probabilities(np.full((3, 12), .4), public_preview=False)


if __name__ == '__main__': unittest.main()
