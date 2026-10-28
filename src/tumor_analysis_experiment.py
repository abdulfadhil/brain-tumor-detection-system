import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model

# Load Trained Model
model = load_model("brain_tumor_cnn.h5", compile=False)

# Tumor Type Mapping
tumor_types = ["Glioma", "Meningioma", "Pituitary"]

# Preprocessing Function (Matches Training Pipeline)
def preprocess_mri(file_path, target_size=(128, 128), depth=64):
    try:
        # Load MRI scan
        img = nib.load(file_path).get_fdata()
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # Normalize

        # Ensure at least 'depth' slices exist
        if img.shape[2] < depth:
            print(f"MRI scan has insufficient slices ({img.shape[2]})!")
            return None, None

        # Corrected Depth Ordering: Select 'depth' slices correctly
        selected_slices = np.linspace(0, img.shape[2] - 1, depth, dtype=int)

        # Resize each selected slice correctly
        resized_slices = np.array([
            cv2.resize(img[:, :, i], target_size) for i in selected_slices
        ], dtype=np.float32)

        # Ensure shape matches training (DEPTH, HEIGHT, WIDTH, CHANNELS)
        resized_slices = np.expand_dims(resized_slices, axis=-1)  # Add channel
        return np.expand_dims(resized_slices, axis=0), img  # Add batch dimension & return original scan

    except Exception as e:
        print(f"Error processing MRI: {e}")
        return None, None

# Function to Predict Tumor, Type, Size & LSG/HSG Classification
def analyze_tumor(mri_path):
    image, original_scan = preprocess_mri(mri_path)
    if image is None:
        return "Error: Invalid MRI file!"

    # Make Prediction
    predictions = model.predict(image)
    tumor_probability = predictions[0][0]  # First neuron → Tumor presence
    tumor_type_idx = np.argmax(predictions[0][1:])  # Remaining neurons → Tumor type

    has_tumor = tumor_probability > 0.5
    tumor_type = tumor_types[tumor_type_idx] if has_tumor else "No Tumor"

    print(f"Tumor probability: {tumor_probability:.4f}")
    print(f"Tumor type: {tumor_type}")

    if not has_tumor:
        return "No Tumor Detected!"

    # Tumor Detected - Extract Details
    tumor_mask = original_scan > np.percentile(original_scan, 95)  # Adaptive thresholding
    tumor_voxels = np.sum(tumor_mask)  # Count tumor pixels

    # Extract voxel spacing from MRI metadata
    voxel_dims = nib.load(mri_path).header.get_zooms()
    voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]  # mm³ per voxel

    tumor_size_mm3 = tumor_voxels * voxel_volume  # Calculate actual size

    # Classify as LSG (Low-Stage Growth) or HSG (High-Stage Growth)
    lsg_hsg = "HSG (High-Stage Growth)" if tumor_size_mm3 > 5000 else "LSG (Low-Stage Growth)"

    # Find the slice with maximum tumor presence
    slice_sums = np.sum(tumor_mask, axis=(0, 1))
    max_tumor_slice = np.argmax(slice_sums)  # Find slice with the most tumor pixels

    return {
        "Prediction": "Tumor Detected",
        "Tumor Probability": f"{tumor_probability:.4f}",
        "Tumor Type": tumor_type,
        "Tumor Size (Voxels)": tumor_voxels,
        "Approx. Tumor Size (mm³)": round(tumor_size_mm3, 2),
        "Growth Stage": lsg_hsg,
        "Most Affected Slice": max_tumor_slice
    }

# Run Prediction (Example)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run experimental analysis on a 3D MRI file."
    )
    parser.add_argument("mri_path", help="Path to the MRI NIfTI file")
    args = parser.parse_args()

    if not os.path.isfile(args.mri_path):
        parser.error("MRI file not found.")

    result = analyze_tumor(args.mri_path)
    print("Experimental MRI analysis:", result)