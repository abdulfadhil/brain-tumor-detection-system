from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import nibabel as nib
import numpy as np

from src.mri_3d_visualization import generate_3d_plot


class MRIVisualizationTests(unittest.TestCase):
    def create_patient(self, folder, constant=False):
        patient = folder.name
        affine = np.eye(4)
        shape = (16, 16, 16)

        for modality in ("t1", "t1ce", "t2", "flair"):
            if constant:
                data = np.ones(shape, dtype=np.float32)
            else:
                data = np.arange(
                    np.prod(shape), dtype=np.float32
                ).reshape(shape)

            image = nib.Nifti1Image(data, affine)
            nib.save(image, str(folder / f"{patient}_{modality}.nii.gz"))

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

            result = generate_3d_plot(str(folder))

            self.assertIsInstance(result, str)
            self.assertIn("<div", result.lower())
            self.assertIn("plotly", result.lower())
            self.assertIn("Brain", result)
            self.assertIn("Tumor", result)

    def test_constant_intensity_volumes_are_rejected(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "patient"
            folder.mkdir()
            self.create_patient(folder, constant=True)

            with self.assertRaisesRegex(
                ValueError, "no usable intensity variation"
            ):
                generate_3d_plot(str(folder))


if __name__ == "__main__":
    unittest.main()
