import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import os

def generate_3d_plot(folder_path):
    # ✅ Detect the correct base name dynamically
    all_files = os.listdir(folder_path)
    base_name = None

    # Find the correct base name
    for file in all_files:
        if file.endswith("_t1.nii.gz"):
            base_name = file.replace("_t1.nii.gz", "")
            break

    if not base_name:
        raise FileNotFoundError(f"Could not detect the correct base name in {folder_path}")

    # ✅ Load MRI Modalities & Tumor Mask
    modalities = ["t1", "t1ce", "t2", "flair"]
    mri_data = []

    for modality in modalities:
        mri_path = os.path.join(folder_path, f"{base_name}_{modality}.nii.gz")
        if os.path.exists(mri_path):
            print(f"✅ Found: {mri_path}")  # Debugging print
            mri_data.append(nib.load(mri_path).get_fdata())
        else:
            raise FileNotFoundError(f"❌ Missing MRI file: {mri_path}")

    # ✅ Merge All Modalities (Average for Better Detail)
    brain_mri = np.mean(mri_data, axis=0)

    # ✅ Load Tumor Segmentation Mask
    tumor_mask_path = os.path.join(folder_path, f"{base_name}_seg.nii.gz")
    if os.path.exists(tumor_mask_path):
        print(f"✅ Found: {tumor_mask_path}")  # Debugging print
        tumor_mask = nib.load(tumor_mask_path).get_fdata()
    else:
        raise FileNotFoundError(f"❌ Missing tumor mask: {tumor_mask_path}")

    # ✅ Ensure Shapes Match
    assert brain_mri.shape == tumor_mask.shape, "Brain and Tumor mask shapes do not match!"

    # ✅ Normalize Brain MRI for Better Visualization
    brain_mri = (brain_mri - np.min(brain_mri)) / (np.max(brain_mri) - np.min(brain_mri))

    # ✅ Get 3D Coordinates for Brain and Tumor
    z_brain, y_brain, x_brain = np.where(brain_mri > 0)  # Include all brain voxels
    brain_intensity = brain_mri[z_brain, y_brain, x_brain]

    z_tumor, y_tumor, x_tumor = np.where(tumor_mask > 0)

    # ✅ Downsample Brain Points to Prevent Plotly Lag
    brain_indices = np.random.choice(len(x_brain), size=min(100000, len(x_brain)), replace=False)
    x_brain, y_brain, z_brain, brain_intensity = x_brain[brain_indices], y_brain[brain_indices], z_brain[brain_indices], brain_intensity[brain_indices]

    # ✅ Create 3D Plot
    fig = go.Figure()

    # ✅ Add Brain (Gray) - **More Visible**
    fig.add_trace(go.Scatter3d(
        x=x_brain, y=y_brain, z=z_brain,
        mode='markers',
        marker=dict(size=2, color=brain_intensity, colorscale="gray", opacity=0.5),
        name="Brain"
    ))

    # ✅ Add Tumor (Red)
    fig.add_trace(go.Scatter3d(
        x=x_tumor, y=y_tumor, z=z_tumor,
        mode='markers',
        marker=dict(size=3, color="red", opacity=1.0),
        name="Tumor"
    ))

    # ✅ Layout & Labels
    fig.update_layout(
        title="🧠 3D Brain MRI with Tumor Visualization",
        scene=dict(
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            zaxis_title="Z-axis"
        )
    )

    # ✅ Convert figure to HTML for Flask
    return fig.to_html(full_html=False)
