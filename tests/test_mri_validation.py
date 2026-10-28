from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import nibabel as nib
import numpy as np

from src.mri_input_validation import validate_patient_folder


class MRIValidationTests(unittest.TestCase):
    def create_patient(self, folder, shape=(8, 8, 8), empty_mask=False):
        patient = folder.name
        affine = np.eye(4)

        for modality in ("t1", "t1ce", "t2", "flair", "seg"):
            if modality == "seg":
                data = np.zeros(shape, dtype=np.float32)
                if not empty_mask:
                    data[4:6, 4:6, 2:4] = 1
            else:
                data = np.arange(
                    np.prod(shape), dtype=np.float32
                ).reshape(shape)

            image = nib.Nifti1Image(data, affine)
            nib.save(image, str(folder / f"{patient}_{modality}.nii.gz"))

    def test_valid_patient_is_accepted(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "patient"
            folder.mkdir()
            self.create_patient(folder)

            result = validate_patient_folder(folder)

            self.assertEqual(result["shape"], (8, 8, 8))
            self.assertEqual(set(result["images"]), {
                "t1", "t1ce", "t2", "flair", "seg"
            })

    def test_missing_file_is_rejected(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "patient"
            folder.mkdir()
            self.create_patient(folder)
            (folder / "patient_t2.nii.gz").unlink()

            with self.assertRaisesRegex(ValueError, "Missing required MRI file"):
                validate_patient_folder(folder)

    def test_mismatched_dimensions_are_rejected(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "patient"
            folder.mkdir()
            self.create_patient(folder)

            mismatch = nib.Nifti1Image(
                np.ones((7, 8, 8), dtype=np.float32), np.eye(4)
            )
            nib.save(mismatch, str(folder / "patient_t2.nii.gz"))

            with self.assertRaisesRegex(ValueError, "dimensions do not match"):
                validate_patient_folder(folder)

    def test_empty_segmentation_mask_is_rejected(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "patient"
            folder.mkdir()
            self.create_patient(folder, empty_mask=True)

            with self.assertRaisesRegex(ValueError, "no positive voxels"):
                validate_patient_folder(folder)


if __name__ == "__main__":
    unittest.main()
