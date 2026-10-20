import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ✅ CHECK & CONFIGURE GPU USAGE
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(f"❌ Error enabling memory growth: {e}")

print("🔍 Available GPUs:", tf.config.experimental.list_logical_devices('GPU'))

# ✅ HYPERPARAMETERS
IMAGE_SIZE = (128, 128)
DEPTH = 64
DATASET_PATH = os.environ.get("BRAIN_TUMOR_DATASET", "").strip()
EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 0.001

# ✅ FUNCTION TO LOAD MRI SCANS
def load_mri_data(dataset_path, num_samples=400):
    images, labels = [], []
    patient_dirs = os.listdir(dataset_path)[:num_samples]

    for patient in patient_dirs:
        patient_path = os.path.join(dataset_path, patient)

        if os.path.isdir(patient_path):
            t1_path = os.path.join(patient_path, f"{patient}_t1.nii.gz")
            seg_path = os.path.join(patient_path, f"{patient}_seg.nii.gz")

            if os.path.exists(t1_path) and os.path.exists(seg_path):
                try:
                    image = nib.load(t1_path).get_fdata()
                    label = nib.load(seg_path).get_fdata()
                except Exception as e:
                    print(f"❌ Error loading {t1_path}: {e}")
                    continue

                if image.shape[2] < DEPTH:
                    continue

                image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize

                resized_slices = np.array([
                    tf.image.resize(np.expand_dims(image[:, :, i], axis=-1), IMAGE_SIZE).numpy()
                    for i in np.linspace(0, image.shape[2] - 1, DEPTH, dtype=int)
                ], dtype=np.float32)

                images.append(resized_slices)
                labels.append(np.max(label) > 0)

    return np.array(images), np.array(labels)

# ✅ CHECK DATASET PATH
if not DATASET_PATH or not os.path.isdir(DATASET_PATH):
    raise SystemExit(
        "Set BRAIN_TUMOR_DATASET to a valid MRI dataset directory."
    )

# ✅ LOAD DATA
X, y = load_mri_data(DATASET_PATH, num_samples=400)
X = np.expand_dims(X, axis=-1)
y = np.array(y).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ BUILD 3D CNN MODEL
def build_3d_cnn():
    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(DEPTH, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
        MaxPooling3D(pool_size=(2, 2, 2)),

        Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ✅ TRAIN MODEL ON GPU
with tf.device('/GPU:0'):
    model = build_3d_cnn()
    model.summary()
    print("🚀 Training Model on GPU...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# SAVE MODEL
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_cnn.h5")
model.save(MODEL_PATH)
print(f"Model saved successfully: {MODEL_PATH}")
