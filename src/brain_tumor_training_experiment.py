import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

# ENABLE GPU MEMORY GROWTH
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# SET DATASET PATH
DATASET_PATH = os.environ.get("BRAIN_TUMOR_DATASET", "").strip()

# IMAGE SETTINGS
IMAGE_SIZE = (128, 128)
DEPTH = 64  # Number of slices

# FUNCTION TO LOAD MRI SCANS
def load_mri_data(dataset_path, num_samples=20):
    images, labels = [], []
    patient_dirs = os.listdir(dataset_path)[:num_samples]

    for patient in patient_dirs:
        patient_path = os.path.join(dataset_path, patient)

        if os.path.isdir(patient_path):
            t1_path = os.path.join(patient_path, f"{patient}_t1.nii.gz")
            seg_path = os.path.join(patient_path, f"{patient}_seg.nii.gz")

            if os.path.exists(t1_path) and os.path.exists(seg_path):
                image = nib.load(t1_path).get_fdata()
                label = nib.load(seg_path).get_fdata()

                # Normalize
                image = (image - np.min(image)) / (np.max(image) - np.min(image))

                # Resize slices
                resized_slices = np.array([
                    tf.image.resize(np.expand_dims(image[:, :, i], axis=-1), IMAGE_SIZE).numpy()
                    for i in np.linspace(0, image.shape[2] - 1, DEPTH, dtype=int)
                ], dtype=np.float32)

                images.append(resized_slices)
                labels.append(np.max(label) > 0)

    return np.array(images), np.array(labels)

# LOAD DATA
if not DATASET_PATH or not os.path.isdir(DATASET_PATH):
    raise SystemExit(
        "Dataset folder not found. Set BRAIN_TUMOR_DATASET to your dataset directory."
    )

X, y = load_mri_data(DATASET_PATH, num_samples=None)

if len(X) == 0:
    raise SystemExit("No valid patient scans found in the configured dataset folder.")
print(f"Loaded {len(X)} MRI Scans, Shape: {X.shape}")

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PREPARE TF.DATA PIPELINE (Fixed Out of Range Issue)
def batch_generator(X, y, batch_size=1):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

train_dataset = tf.data.Dataset.from_generator(
    lambda: batch_generator(X_train, y_train, batch_size=1),
    output_signature=(tf.TensorSpec(shape=(None, DEPTH, 128, 128, 1), dtype=tf.float32),
                      tf.TensorSpec(shape=(None,), dtype=tf.int32))
).repeat().prefetch(tf.data.AUTOTUNE)  # FIXED: .repeat() to prevent OUT_OF_RANGE

test_dataset = tf.data.Dataset.from_generator(
    lambda: batch_generator(X_test, y_test, batch_size=1),
    output_signature=(tf.TensorSpec(shape=(None, DEPTH, 128, 128, 1), dtype=tf.float32),
                      tf.TensorSpec(shape=(None,), dtype=tf.int32))
).repeat().prefetch(tf.data.AUTOTUNE)  # FIXED: .repeat() to prevent OUT_OF_RANGE

# BUILD 3D CNN MODEL (Fixed input_shape issue)
model = Sequential([
    Input(shape=(DEPTH, 128, 128, 1)),  # FIXED: Use explicit Input() instead of input_shape in Conv3D

    Conv3D(16, kernel_size=(3,3,3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
    MaxPooling3D(pool_size=(2,2,2)),

    Conv3D(32, kernel_size=(3,3,3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
    MaxPooling3D(pool_size=(2,2,2)),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # L2 Regularization
    Dropout(0.4),  # FIXED: Increased dropout to prevent overfitting
    Dense(1, activation='sigmoid')
])

# COMPILE MODEL
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# TRAIN MODEL (Using fit instead of train_on_batch)
epochs = 5  # More epochs for better learning
steps_per_epoch = len(X_train) // 1  # Batch size = 1

model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)

# FINAL EVALUATION
test_loss, test_acc = model.evaluate(test_dataset, steps=len(X_test))
print(f"Test accuracy: {test_acc:.4f}")
MODEL_OUTPUT_PATH = os.environ.get("BRAIN_TUMOR_MODEL_OUTPUT", "brain_tumor_3dcnn.h5")
model.save(MODEL_OUTPUT_PATH)
