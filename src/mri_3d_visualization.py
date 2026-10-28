from pathlib import Path

import nibabel as nib
import numpy as np
import plotly.graph_objects as go


MODALITIES = ("t1", "t1ce", "t2", "flair")


def _select_point_indices(point_count, max_points):
    """Return deterministic indices capped at max_points."""
    if point_count <= max_points:
        return np.arange(point_count)
    return np.linspace(0, point_count - 1, max_points, dtype=np.int64)


def generate_3d_plot(folder_path):
    """Generate a reproducible 3D plot from MRI volumes and a supplied mask."""
    folder = Path(folder_path)
    t1_files = sorted(folder.glob("*_t1.nii.gz"))
    if len(t1_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_t1.nii.gz file in {folder}."
        )

    patient = t1_files[0].name.removesuffix("_t1.nii.gz")
    mri_volumes = []
    reference_image = None

    for modality in MODALITIES:
        path = folder / f"{patient}_{modality}.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing MRI file: {path.name}")

        image = nib.load(str(path))
        if len(image.shape) != 3:
            raise ValueError(f"{path.name} must contain a 3D volume.")

        if reference_image is None:
            reference_image = image
        else:
            if image.shape != reference_image.shape:
                raise ValueError(
                    f"MRI dimensions do not match: {path.name}."
                )
            if not np.allclose(
                image.affine, reference_image.affine,
                rtol=1e-4, atol=1e-3
            ):
                raise ValueError(
                    f"Spatial orientation or position does not match: "
                    f"{path.name}."
                )

        volume = image.get_fdata(dtype=np.float32)
        if not np.isfinite(volume).all():
            raise ValueError(f"{path.name} contains invalid voxel values.")
        mri_volumes.append(volume)

    brain_mri = np.mean(np.stack(mri_volumes, axis=0), axis=0)

    mask_path = folder / f"{patient}_seg.nii.gz"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing segmentation mask: {mask_path.name}")

    mask_image = nib.load(str(mask_path))
    if len(mask_image.shape) != 3:
        raise ValueError("Segmentation mask must contain a 3D volume.")

    if not np.allclose(
        mask_image.affine, reference_image.affine,
        rtol=1e-4, atol=1e-3
    ):
        raise ValueError(
            "Spatial orientation or position does not match: "
            f"{mask_path.name}."
        )

    tumor_mask = mask_image.get_fdata(dtype=np.float32)

    if brain_mri.shape != tumor_mask.shape:
        raise ValueError("MRI and segmentation mask dimensions do not match.")
    if not np.isfinite(tumor_mask).all():
        raise ValueError("Segmentation mask contains invalid voxel values.")

    intensity_min = float(brain_mri.min())
    intensity_max = float(brain_mri.max())
    if intensity_max <= intensity_min:
        raise ValueError("MRI volumes have no usable intensity variation.")

    brain_mri = (brain_mri - intensity_min) / (intensity_max - intensity_min)

    x_brain, y_brain, z_brain = np.where(brain_mri > 0)
    if x_brain.size == 0:
        raise ValueError("MRI volumes contain no usable brain points.")

    brain_intensity = brain_mri[x_brain, y_brain, z_brain]
    x_tumor, y_tumor, z_tumor = np.where(tumor_mask > 0)

    max_brain_points = 100_000
    selected = _select_point_indices(x_brain.size, max_brain_points)
    x_brain = x_brain[selected]
    y_brain = y_brain[selected]
    z_brain = z_brain[selected]
    brain_intensity = brain_intensity[selected]

    max_mask_points = 50_000
    selected = _select_point_indices(x_tumor.size, max_mask_points)
    x_tumor = x_tumor[selected]
    y_tumor = y_tumor[selected]
    z_tumor = z_tumor[selected]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=x_brain,
            y=y_brain,
            z=z_brain,
            mode="markers",
            marker=dict(
                size=2, color=brain_intensity, colorscale="gray", opacity=0.5
            ),
            name="Brain MRI",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x_tumor,
            y=y_tumor,
            z=z_tumor,
            mode="markers",
            marker=dict(size=3, color="red", opacity=1.0),
            name="Supplied segmentation mask",
        )
    )

    fig.update_layout(
        title="3D Brain MRI Visualization",
        scene=dict(
            xaxis_title="Voxel X",
            yaxis_title="Voxel Y",
            zaxis_title="Voxel Z",
        ),
    )
    return fig.to_html(full_html=False)
