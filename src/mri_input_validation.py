from pathlib import Path

import nibabel as nib
import numpy as np


MODALITIES = ("t1", "t1ce", "t2", "flair", "seg")


def validate_patient_folder(folder):
    """Validate the five NIfTI files for one patient before analysis."""
    folder = Path(folder)
    patient = folder.name
    images = {}
    reference_shape = None
    reference_affine = None
    reference_spacing = None

    for modality in MODALITIES:
        path = folder / f"{patient}_{modality}.nii.gz"
        if not path.is_file():
            raise ValueError(f"Missing required MRI file: {path.name}")

        try:
            image = nib.load(str(path))
            shape = tuple(image.shape)
            affine = np.asarray(image.affine, dtype=np.float64)
            spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
        except Exception as exc:
            raise ValueError(
                f"Could not read {path.name} as a valid NIfTI file."
            ) from exc

        if len(shape) != 3 or any(size <= 0 for size in shape):
            raise ValueError(f"{path.name} must contain a 3D volume.")

        if not np.isfinite(affine).all() or not np.isfinite(spacing).all():
            raise ValueError(f"{path.name} has invalid spatial metadata.")

        if any(value <= 0 for value in spacing):
            raise ValueError(f"{path.name} has invalid voxel spacing.")

        if reference_shape is None:
            reference_shape = shape
            reference_affine = affine
            reference_spacing = spacing
        else:
            if shape != reference_shape:
                raise ValueError(
                    f"MRI dimensions do not match: {path.name} has {shape}, "
                    f"expected {reference_shape}."
                )

            if not np.allclose(
                spacing, reference_spacing, rtol=1e-4, atol=1e-4
            ):
                raise ValueError(
                    f"Voxel spacing does not match across files: {path.name}."
                )

            if not np.allclose(
                affine, reference_affine, rtol=1e-4, atol=1e-3
            ):
                raise ValueError(
                    f"Spatial orientation or position does not match: "
                    f"{path.name}."
                )

        images[modality] = image

    # Check image data without retaining all five full volumes in memory.
    for modality in ("t1", "t1ce", "t2", "flair"):
        data = images[modality].dataobj
        sample = np.asarray(data[::8, ::8, ::4], dtype=np.float32)
        if sample.size == 0 or not np.isfinite(sample).all():
            raise ValueError(
                f"{modality.upper()} contains invalid sampled voxel values."
            )
        if float(sample.max()) <= float(sample.min()):
            raise ValueError(
                f"{modality.upper()} appears to have constant intensity."
            )

    mask_sample = np.asarray(
        images["seg"].dataobj[::4, ::4, ::2], dtype=np.float32
    )
    if mask_sample.size == 0 or not np.isfinite(mask_sample).all():
        raise ValueError("Segmentation mask contains invalid voxel values.")

    if not np.any(mask_sample > 0):
        raise ValueError("Segmentation mask contains no positive voxels.")

    return {
        "shape": reference_shape,
        "spacing_mm": reference_spacing,
        "images": images,
    }
