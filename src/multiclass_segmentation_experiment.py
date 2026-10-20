import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# ✅ Set Path to Data
base_path = os.environ.get("BRAIN_TUMOR_SAMPLE_DIR", "").strip()
if not base_path:
    raise SystemExit(
        "Set BRAIN_TUMOR_SAMPLE_DIR to the directory containing the MRI files."
    )

base_path = os.path.abspath(base_path)

modalities = ["t1", "t1ce", "t2", "flair"]
input_shape = (128, 128, 128)  # Resize images to this shape

# ✅ Load and Preprocess MRI Scans
def load_mri_scans(base_path, modalities):
    mri_data = []
    for modality in modalities:
        mri_path = os.path.join(base_path, f"BraTS2021_00056_{modality}.nii.gz")
        img = nib.load(mri_path).get_fdata()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize
        img = tf.image.resize(img, input_shape)  # Resize
        mri_data.append(img)
    return np.stack(mri_data, axis=-1)  # Shape: (128, 128, 128, 4)

# ✅ Load Tumor Mask
def load_tumor_mask(base_path):
    mask_path = os.path.join(base_path, "BraTS2021_00056_seg.nii.gz")
    mask = nib.load(mask_path).get_fdata()
    mask = tf.image.resize(mask, input_shape)  # Resize
    mask = np.round(mask).astype(int)  # Ensure integers (0, 1, 2, 3, 4)
    return to_categorical(mask, num_classes=5)  # Convert to categorical labels

# ✅ Load Data
X_data = load_mri_scans(base_path, modalities)  # Shape: (128, 128, 128, 4)
Y_data = load_tumor_mask(base_path)  # Shape: (128, 128, 128, 5)

# ✅ Handle Dataset Splitting
if len(X_data.shape) == 4:  # If only one patient, expand dimensions
    X_data = np.expand_dims(X_data, axis=0)
    Y_data = np.expand_dims(Y_data, axis=0)

if X_data.shape[0] > 1:
    X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
else:
    X_train, Y_train = X_data, Y_data
    X_val, Y_val = X_data, Y_data

# ✅ Build 3D CNN Model
model = Sequential([
    Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(128, 128, 128, 4)),
    MaxPooling3D(pool_size=(2, 2, 2)),

    Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 2)),

    Conv3D(128, kernel_size=(3, 3, 3), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 Classes (0, 1, 2, 3, 4)
])

# ✅ Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train Model
history = model.fit(X_train, Y_train, epochs=20, batch_size=2, validation_data=(X_val, Y_val))

# ✅ Save Model
model.save("brain_tumor_3dcnn.h5")

print("✅ Training Complete!")
