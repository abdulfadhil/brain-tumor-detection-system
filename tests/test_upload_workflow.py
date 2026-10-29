from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

import app as application


class UploadWorkflowTests(unittest.TestCase):
    def make_nifti_bytes(self, data, suffix=".nii.gz"):
        with TemporaryDirectory() as temp:
            path = Path(temp) / f"synthetic{suffix}"
            image = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
            nib.save(image, str(path))
            return path.read_bytes()

    def make_uploads(self, suffix=".nii.gz"):
        patient = "SyntheticPatient"
        shape = (16, 16, 16)
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        mask = np.zeros(shape, dtype=np.float32)
        mask[4:8, 4:8, 4:8] = 1

        return [
            (
                BytesIO(
                    self.make_nifti_bytes(
                        mask if modality == "seg" else data, suffix
                    )
                ),
                f"{patient}_{modality}{suffix}",
            )
            for modality in ("t1", "t1ce", "t2", "flair", "seg")
        ]

    def setUp(self):
        application.app.config.update(TESTING=True)
        self.client = application.app.test_client()

    def test_identifies_nii_and_nii_gz(self):
        expected = ("Patient_001", "t1ce")
        self.assertEqual(
            application.identify_upload("Patient_001_t1ce.nii"), expected
        )
        self.assertEqual(
            application.identify_upload("Patient_001_t1ce.nii.gz"), expected
        )

    def test_identifies_windows_browser_fakepath(self):
        expected = ("Patient_001", "t1")
        self.assertEqual(
            application.identify_upload(
                r"C:\fakepath\Patient_001_t1.nii.gz"
            ),
            expected,
        )

    def test_identifies_forward_slash_path(self):
        expected = ("Patient_001", "t1")
        self.assertEqual(
            application.identify_upload(
                "C:/fakepath/Patient_001_t1.nii.gz"
            ),
            expected,
        )

    def test_rejects_unrecognized_filename(self):
        self.assertIsNone(application.identify_upload("notes.txt"))

    def test_rejects_missing_uploads(self):
        response = self.client.post("/upload", data={})
        self.assertEqual(response.status_code, 400)

    def test_rejects_missing_modality(self):
        response = self.client.post(
            "/upload",
            data={"file": self.make_uploads()[:-1]},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Missing required files", response.data)

    def test_rejects_duplicate_modality(self):
        uploads = self.make_uploads()
        uploads.append(
            (
                BytesIO(self.make_nifti_bytes(np.ones((16, 16, 16)))),
                "SyntheticPatient_t1.nii.gz",
            )
        )
        response = self.client.post(
            "/upload",
            data={"file": uploads},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Duplicate", response.data)

    def test_accepts_five_valid_compressed_files(self):
        results = {
            "Prediction": "Experimental CNN output — not a diagnosis",
            "Tumor Probability": "Not reported",
            "Tumor Size (Voxels)": "64",
            "Approx. Tumor Size (mm³)": "64.00",
            "Most Affected Slice": "Not calculated",
        }
        with patch.object(
            application, "analyze_patient",
            return_value=("<div>synthetic visualization</div>", results, None),
        ):
            response = self.client.post(
                "/upload",
                data={"file": self.make_uploads(".nii.gz")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"synthetic visualization", response.data)

    def test_accepts_five_uncompressed_files(self):
        results = {
            "Prediction": "Experimental CNN output — not a diagnosis",
            "Tumor Probability": "Not reported",
            "Tumor Size (Voxels)": "64",
            "Approx. Tumor Size (mm³)": "64.00",
            "Most Affected Slice": "Not calculated",
        }
        with patch.object(
            application, "analyze_patient",
            return_value=("<div>synthetic visualization</div>", results, None),
        ):
            response = self.client.post(
                "/upload",
                data={"file": self.make_uploads(".nii")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"synthetic visualization", response.data)


if __name__ == "__main__":
    unittest.main()
