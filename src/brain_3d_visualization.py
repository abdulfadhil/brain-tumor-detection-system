import nibabel as nib
import numpy as np
import plotly.graph_objects as go

# ✅ Load MRI Modalities & Tumor Mask
import os

base_path = os.environ.get("BRAIN_TUMOR_SAMPLE_DIR", "").strip()
if not base_path:
    raise SystemExit(
        "Set BRAIN_TUMOR_SAMPLE_DIR to the directory containing the MRI files."
    )

base_path = os.path.abspath(base_path)
modalities = ["t1", "t1ce", "t2", "flair"]
mri_data = []

for modality in modalities:
    mri_path = f"{base_path}BraTS2021_00056_{modality}.nii.gz"
    mri_data.append(nib.load(mri_path).get_fdata())

# ✅ Merge All Modalities (Average for Better Detail)
brain_mri = np.mean(mri_data, axis=0)

# ✅ Load Tumor Segmentation Mask
tumor_mask_path = f"{base_path}BraTS2021_00056_seg.nii.gz"
tumor_mask = nib.load(tumor_mask_path).get_fdata()

# ✅ Normalize Brain MRI - Increase Contrast
brain_mri = (brain_mri - np.percentile(brain_mri, 5)) / (np.percentile(brain_mri, 95) - np.percentile(brain_mri, 5))
brain_mri = np.clip(brain_mri, 0, 1)  # Keep values between 0 and 1

# ✅ Get 3D Coordinates for Brain and Tumor
z_brain, y_brain, x_brain = np.where(brain_mri > 0.1)  # Lowered threshold
brain_intensity = brain_mri[z_brain, y_brain, x_brain]

z_tumor, y_tumor, x_tumor = np.where(tumor_mask > 0)

# ✅ Debugging: Check MRI Shape
print("Brain MRI Shape:", brain_mri.shape)
print("Tumor Mask Shape:", tumor_mask.shape)

# ✅ Create 3D Plot
fig = go.Figure()

# ✅ Add Brain (Gray) - Increase Opacity & Size
fig.add_trace(go.Scatter3d(
    x=x_brain, y=y_brain, z=z_brain,
    mode='markers',
    marker=dict(size=2, color=brain_intensity, colorscale="gray", opacity=0.8),
    name="Brain"
))

# ✅ Add Tumor (Red) - Make it Semi-Transparent
fig.add_trace(go.Scatter3d(
    x=x_tumor, y=y_tumor, z=z_tumor,
    mode='markers',
    marker=dict(size=3, color="red", opacity=0.6),
    name="Tumor"
))

# ✅ Layout & Labels
fig.update_layout(
    title="3D Brain MRI with Tumor Visualization",
    scene=dict(
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
        zaxis_title="Z-axis"
    )
)

# ✅ Show Interactive 3D Model
fig.show()
