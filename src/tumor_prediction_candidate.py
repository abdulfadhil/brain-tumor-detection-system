import argparse
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "brain_tumor_cnn.h5"


def preprocess_mri(file_path, target_size=(128, 128), depth=64):
    """Load an MRI volume and prepare it for the recovered binary CNN."""
    path = Path(file_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"MRI file not found: {path}")

    try:
        nii = nib.load(str(path))
        img = nii.get_fdata(dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Could not read the MRI file: {exc}") from exc

    if img.ndim != 3 or any(size < 1 for size in img.shape):
        raise ValueError("Expected a non-empty 3D MRI volume.")

    if not np.isfinite(img).all():
        raise ValueError("MRI volume contains non-finite voxel values.")

    minimum = float(np.min(img))
    maximum = float(np.max(img))

    if maximum <= minimum:
        raise ValueError("MRI volume has constant intensity; normalization is undefined.")

    normalized = (img - minimum) / (maximum - minimum)
    slice_indices = np.linspace(0, img.shape[2] - 1, depth, dtype=int)

    resized_slices = np.asarray(
        [
            cv2.resize(normalized[:, :, index], target_size)
            for index in slice_indices
        ],
        dtype=np.float32,
    )

    model_input = resized_slices[..., np.newaxis][np.newaxis, ...]
    return model_input, img, nii.header.get_zooms()


def analyze_tumor(mri_path, model=None):
    """Return an experimental model score and a separate intensity estimate."""
    image, original_scan, voxel_dims = preprocess_mri(mri_path)

    if model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        model = load_model(str(MODEL_PATH), compile=False)

    prediction = np.asarray(model.predict(image, verbose=0))

    if prediction.size != 1:
        raise ValueError(
            f"Expected one binary model output; received shape {prediction.shape}."
        )

    score = float(prediction.reshape(-1)[0])

    if not np.isfinite(score):
        raise ValueError("Model returned a non-finite score.")

    # This is an intensity-based estimate, NOT a CNN-generated tumor mask.
    intensity_threshold = float(np.percentile(original_scan, 95))
    estimated_mask = original_scan > intensity_threshold
    estimated_voxels = int(np.count_nonzero(estimated_mask))

    if len(voxel_dims) < 3:
        raise ValueError("MRI metadata does not contain three voxel dimensions.")

    voxel_volume = float(np.prod(voxel_dims[:3]))

    if not np.isfinite(voxel_volume) or voxel_volume <= 0:
        raise ValueError("MRI metadata contains invalid voxel spacing.")

    estimated_volume_mm3 = estimated_voxels * voxel_volume
    slice_counts = np.count_nonzero(estimated_mask, axis=(0, 1))
    most_intense_slice = int(np.argmax(slice_counts))

    return {
        "Experimental model score (uncalibrated)": round(score, 6),
        "Threshold-based estimated voxels (not segmentation)": estimated_voxels,
        "Threshold-based estimated volume (mm^3, not tumor measurement)": round(
            estimated_volume_mm3, 2
        ),
        "Slice with most thresholded voxels (not tumor localization)": (
            most_intense_slice
        ),
        "Disclaimer": (
            "Research use only. This score is uncalibrated, and the intensity "
            "threshold is not a tumor segmentation or a medical diagnosis."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run an experimental binary CNN on a 3D MRI volume."
    )
    parser.add_argument("mri_path", help="Path to a 3D MRI file, such as .nii.gz")
    args = parser.parse_args()

    try:
        result = analyze_tumor(args.mri_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print("Experimental MRI analysis:")
    for label, value in result.items():
        print(f"{label}: {value}")


if __name__ == "__main__":
    main()
