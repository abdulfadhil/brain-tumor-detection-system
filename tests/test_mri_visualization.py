from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import nibabel as nib
import numpy as np

from src.mri_3d_visualization import (
    _select_point_indices,
    generate_3d_plot,
)

class MRIVisualizationTests(unittest.TestCase):

    def create_patient(self, folder, constant=False, shape=(16, 16, 16)):
        patient = folder.name
        affine = np.eye(4)

        for modality in ("t1", "t1ce", "t2", "flair"):
            if constant:
                data = np.ones(shape, dtype=np.float32)
            else:
                data = np.arange(
                    np.prod(shape), dtype=np.float32
                ).reshape(shape)

            nib.save(
                nib.Nifti1Image(data, affine),
                str(folder / f"{patient}_{modality}.nii.gz"),
            )

        mask = np.zeros(shape, dtype=np.float32)
        mask[4:8, 4:8, 4:8] = 1
        nib.save(
            nib.Nifti1Image(mask, affine),
            str(folder / f"{patient}_seg.nii.gz"),
        )

    def test_valid_volumes_produce_plotly_html(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "patient"
            folder.mkdir()
            self.create_patient(folder)

            result = generate_3d_plot(folder)

            self.assertIsInstance(result, str)
            self.assertIn("<div", result.lower())
            self.assertIn("plotly", result.lower())
            self.assertIn("Brain MRI", result)
            self.assertIn("Supplied segmentation mask", result)

    def test_constant_intensity_volumes_are_rejected(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "patient"
            folder.mkdir()
            self.create_patient(folder, constant=True)

            with self.assertRaisesRegex(
                ValueError, "no usable intensity variation"
            ):
                generate_3d_plot(folder)

    def test_mismatched_mask_dimensions_are_rejected(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "patient"
            folder.mkdir()
            self.create_patient(folder)

            mask_path = folder / "patient_seg.nii.gz"
            nib.save(
                nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)),
                str(mask_path),
            )

            with self.assertRaisesRegex(
                ValueError, "dimensions do not match"
            ):
                generate_3d_plot(folder)

    def test_mismatched_mask_affine_is_rejected(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "patient"
            folder.mkdir()
            self.create_patient(folder)

            mask_path = folder / "patient_seg.nii.gz"
            mask = np.zeros((16, 16, 16), dtype=np.float32)
            mask[4:8, 4:8, 4:8] = 1
            affine = np.eye(4)
            affine[0, 3] = 10
            nib.save(nib.Nifti1Image(mask, affine), str(mask_path))

            with self.assertRaisesRegex(
                ValueError, "Spatial orientation or position does not match"
            ):
                generate_3d_plot(folder)

    def test_point_selection_respects_limit(self):
        indices = _select_point_indices(100_000, 50_000)

        self.assertEqual(len(indices), 50_000)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 99_999)
        self.assertEqual(len(np.unique(indices)), 50_000)

    def test_mask_points_are_downsampled(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "patient"
            folder.mkdir()
            self.create_patient(folder, shape=(40, 40, 40))

            mask_path = folder / "patient_seg.nii.gz"
            mask = np.ones((40, 40, 40), dtype=np.float32)
            nib.save(
                nib.Nifti1Image(mask, np.eye(4)), str(mask_path)
            )

            result = generate_3d_plot(folder)
            self.assertIsInstance(result, str)
            self.assertIn("Supplied segmentation mask", result)

    def test_non_finite_mask_is_rejected(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "patient"
            folder.mkdir()
            self.create_patient(folder)

            mask_path = folder / "patient_seg.nii.gz"
            mask = np.zeros((16, 16, 16), dtype=np.float32)
            mask[3, 3, 3] = np.nan
            nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))

            with self.assertRaisesRegex(ValueError, "invalid voxel values"):
                generate_3d_plot(folder)

if __name__ == "__main__":
    unittest.main()
