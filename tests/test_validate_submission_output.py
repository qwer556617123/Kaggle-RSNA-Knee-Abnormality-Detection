import csv
import tempfile
import unittest
from pathlib import Path

from tools.validate_submission_output import TARGETS, validate


class SubmissionOutputValidationTests(unittest.TestCase):
    def test_valid_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["StudyInstanceUID", *TARGETS])
                writer.writeheader()
                for index in range(3):
                    writer.writerow({"StudyInstanceUID": str(index), **{target: (index + 1) / 4 for target in TARGETS}})
            self.assertEqual(validate(Path(directory))["studies"], 3)

    def test_rejects_constant_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["StudyInstanceUID", *TARGETS])
                writer.writeheader()
                for index in range(3):
                    writer.writerow({"StudyInstanceUID": str(index), **{target: 0.5 for target in TARGETS}})
            with self.assertRaisesRegex(ValueError, "globally constant"):
                validate(Path(directory))


if __name__ == "__main__":
    unittest.main()
