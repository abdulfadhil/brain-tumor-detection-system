from pathlib import Path
from tempfile import TemporaryDirectory
import re

import nibabel as nib
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.mri_3d_visualization import generate_3d_plot
from src.mri_input_validation import validate_patient_folder

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "brain_tumor_cnn.h5"
MODALITIES = ("t1", "t1ce", "t2", "flair", "seg")
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError("The recovered CNN model is missing.")
        _model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    return _model



def identify_upload(filename):
    normalized = filename.replace("\\", "/")
    basename = secure_filename(normalized.split("/")[-1])

    match = re.fullmatch(
        r"(.+?)_(t1ce|flair|t1|t2|seg)\.nii(?:\.gz)?",
        basename,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    patient = match.group(1)
    modality = match.group(2).lower()

    if not patient:
        return None

    return patient, modality



def preprocess_t1(path):
    volume = nib.load(str(path)).get_fdata(dtype=np.float32)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("The T1 MRI must be a finite 3D volume.")

    low = float(volume.min())
    high = float(volume.max())
    if high <= low:
        raise ValueError("The T1 MRI has no usable intensity variation.")

    volume = (volume - low) / (high - low)
    indices = np.linspace(0, volume.shape[2] - 1, 64, dtype=int)
    slices = []
    for index in indices:
        image = tf.convert_to_tensor(volume[:, :, index, None])
        image = tf.image.resize(image, (128, 128))
        slices.append(image.numpy())
    return np.expand_dims(np.stack(slices), axis=0)


def analyze_patient(folder):
    validation = validate_patient_folder(folder)
    graph_html = generate_3d_plot(str(folder))

    mask_image = validation["images"]["seg"]
    mask = mask_image.get_fdata(dtype=np.float32)
    if not np.isfinite(mask).all():
        raise ValueError("The segmentation mask contains invalid voxel values.")

    voxel_count = int(np.count_nonzero(mask > 0))
    voxel_volume = float(np.prod(validation["spacing_mm"]))
    volume_mm3 = voxel_count * voxel_volume

    results = {
        "Prediction": "Experimental CNN output — not a diagnosis",
        "Tumor Probability": "Not reported: model is not clinically validated",
        "Tumor Size (Voxels)": f"{voxel_count:,} non-zero voxels in supplied mask",
        "Approx. Tumor Size (mm³)": f"{volume_mm3:,.2f} mm³ in supplied mask",
        "Most Affected Slice": "Not calculated",
    }

    raw_score = None
    if MODEL_PATH.is_file():
        try:
            model_input = preprocess_t1(folder / f"{folder.name}_t1.nii.gz")
            prediction = np.asarray(get_model().predict(model_input, verbose=0))
            if prediction.size != 1:
                raise ValueError("The recovered model did not return one binary score.")
            score = float(prediction.reshape(-1)[0])
            if not np.isfinite(score):
                raise ValueError("The recovered model returned a non-finite score.")
            raw_score = score
        except Exception:
            app.logger.exception(
                "Optional CNN inference failed; showing mask results only."
            )

    return graph_html, results, raw_score


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/upload")
def upload():
    uploads = request.files.getlist("file")
    if not uploads or not any(item.filename for item in uploads):
        return render_template(
            "index.html", error="Please select one patient's five NIfTI files."
        ), 400

    groups = {}
    for item in uploads:
        if not item.filename:
            continue
        identified = identify_upload(item.filename)
        if identified is None:
            continue

        patient, modality = identified
        groups.setdefault(patient, {})
        if modality in groups[patient]:
            return render_template(
                "index.html",
                error=f"Duplicate {modality} file for patient {patient}.",
            ), 400
        groups[patient][modality] = item

    if len(groups) != 1:
        return render_template(
            "index.html",
            error="Upload exactly one patient's five NIfTI files at a time.",
        ), 400

    patient, files = next(iter(groups.items()))
    missing = set(MODALITIES) - set(files)
    if missing:
        return render_template(
            "index.html",
            error="Missing required files: " + ", ".join(sorted(missing)),
        ), 400

    try:
        with TemporaryDirectory(prefix="brain_mri_") as temporary:
            patient_dir = Path(temporary) / patient
            patient_dir.mkdir()

            for modality, item in files.items():
                destination = patient_dir / f"{patient}_{modality}.nii.gz"
                original_name = item.filename.lower()
                if original_name.endswith(".nii.gz"):
                    item.save(destination)
                else:
                    uncompressed = patient_dir / f"{patient}_{modality}.nii"
                    item.save(uncompressed)
                    try:
                        image = nib.load(str(uncompressed))
                        nib.save(image, str(destination))
                    finally:
                        uncompressed.unlink(missing_ok=True)

            graph_html, results, raw_score = analyze_patient(patient_dir)
            return render_template(
                "output.html",
                graph_html=graph_html,
                tumor_results=results,
                experimental_score=(
                    f"{raw_score:.6f}" if raw_score is not None else None
                ),
            )

    except (ValueError, FileNotFoundError) as exc:
        return render_template("index.html", error=str(exc)), 400
    except Exception:
        app.logger.exception("MRI processing failed")
        return render_template(
            "index.html",
            error="Processing failed. Check the server terminal for details.",
        ), 500


@app.errorhandler(413)
def upload_too_large(_error):
    return render_template(
        "index.html",
        error="Upload too large. The combined upload limit is 200 MB.",
    ), 413


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
