import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import cv2

from tensorflow.keras.models import load_model

# Load Trained Model (Avoid Compilation Warning)
model = load_model("brain_tumor_cnn.h5", compile=False)

# Define MRI Preprocessing Function
def preprocess_mri(file_path, target_size=(128, 128), depth=64):
    try:
        # Load MRI scan
        img = nib.load(file_path).get_fdata()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize

        # Resize slices
        resized_slices = np.array([
            cv2.resize(img[:, :, i], target_size)
            for i in np.linspace(0, img.shape[2] - 1, depth, dtype=int)
        ], dtype=np.float32)

        # Add channel and batch dimension
        resized_slices = np.expand_dims(resized_slices, axis=-1)  # Add channel
        return np.expand_dims(resized_slices, axis=0), img  # Add batch dimension & return original scan

    except Exception as e:
        print(f"Error processing MRI: {e}")
        return None, None

# Function to Predict Tumor and Extract Size & Location
def analyze_tumor(mri_path):
    image, original_scan = preprocess_mri(mri_path)
    if image is None:
        return "Error: Invalid MRI file!"

    # Make Prediction
    prediction_prob = model.predict(image)[0][0]
    has_tumor = prediction_prob > 0.5

    print(f"Tumor probability: {prediction_prob:.4f}")

    if not has_tumor:
        return "No Tumor Detected!"

    # Tumor Detected - Extract Details
    tumor_mask = original_scan > np.percentile(original_scan, 95)  # Adaptive thresholding
    tumor_voxels = np.sum(tumor_mask)  # Count tumor pixels

    # Extract voxel spacing from MRI metadata
    voxel_dims = nib.load(mri_path).header.get_zooms()
    voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]  # mm³ per voxel

    tumor_size_mm3 = tumor_voxels * voxel_volume  # Calculate actual size

    # Find the slice with maximum tumor presence
    slice_sums = np.sum(tumor_mask, axis=(0, 1))
    max_tumor_slice = np.argmax(slice_sums)  # Find slice with the most tumor pixels

    return {
        "Prediction": "Tumor Detected",
        "Tumor Probability": f"{prediction_prob:.4f}",
        "Tumor Size (Voxels)": tumor_voxels,
        "Approx. Tumor Size (mm³)": round(tumor_size_mm3, 2),
        "Most Affected Slice": max_tumor_slice
    }

# Run Prediction (Example)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze a 3D NIfTI MRI file for experimental research."
    )
    parser.add_argument("mri_path", help="Path to the MRI NIfTI file")
    args = parser.parse_args()

    if not os.path.isfile(args.mri_path):
        parser.error("MRI file not found.")

    result = analyze_tumor(args.mri_path)
    print("Experimental MRI analysis:", result)