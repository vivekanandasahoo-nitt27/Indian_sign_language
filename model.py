# ============================================================
# INDIAN SIGN LANGUAGE - 229 CLASS MODEL
#
# TRAINING:
#   ALL samples from D:\indian_sign_landmarks
#   85% TRAIN + 15% VALIDATION
#
# FINAL TEST:
#   Separate image dataset:
#   D:\indian_sign_merged_test
#
# The external test images are converted to the same 135
# MediaPipe features before evaluation.
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import json
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(r"D:\indian_sign_landmarks")
TEST_IMAGE_DIR = Path(r"D:\indian_sign_merged_test")
MODEL_DIR = Path(r"D:\indian_sign_model")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

# ALL augmented landmark samples are used.
# Only TRAIN and VALIDATION are split from them.
VALIDATION_SIZE = 0.15

EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# Reduced dropout
DROPOUT_1 = 0.20
DROPOUT_2 = 0.15
DROPOUT_3 = 0.10
DROPOUT_4 = 0.10

# MediaPipe settings
HAND_DETECTION_CONFIDENCE = 0.50
POSE_DETECTION_CONFIDENCE = 0.20
POSE_TRACKING_CONFIDENCE = 0.20
POSE_VISIBILITY_THRESHOLD = 0.20

# Same fallback shoulders used in the landmark pipeline
DEFAULT_LEFT_SHOULDER = np.array(
    [0.69371974, 0.36440188, -0.13180971],
    dtype=np.float32
)

DEFAULT_RIGHT_SHOULDER = np.array(
    [0.31984484, 0.37018980, -0.17071517],
    dtype=np.float32
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# GPU CHECK
# ============================================================

print("\n" + "=" * 75)
print("GPU CHECK")
print("=" * 75)

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
print("Detected GPUs:", gpus)

if not gpus:
    print("\nWARNING: No GPU detected. Training will use CPU.")
else:
    print("\nGPU AVAILABLE")
    for gpu in gpus:
        print("GPU:", gpu)
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


# ============================================================
# LOAD ALL AUGMENTED LANDMARK DATA
# ============================================================

print("\n" + "=" * 75)
print("LOADING ALL AUGMENTED LANDMARK DATA")
print("=" * 75)

X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"
CLASSES_PATH = DATA_DIR / "classes.npy"

for path in [X_PATH, Y_PATH, CLASSES_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

X = np.load(X_PATH)
y = np.load(Y_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

NUM_SAMPLES = X.shape[0]
NUM_FEATURES = X.shape[1]
NUM_CLASSES = len(classes)

print("Total augmented samples:", NUM_SAMPLES)
print("Features:", NUM_FEATURES)
print("Classes:", NUM_CLASSES)


# ============================================================
# VERIFY DATA
# ============================================================

if NUM_FEATURES != 135:
    raise ValueError(
        f"Expected 135 features, but found {NUM_FEATURES}."
    )

if len(X) != len(y):
    raise ValueError("X and y have different numbers of samples.")

if np.min(y) < 0 or np.max(y) >= NUM_CLASSES:
    raise ValueError("Invalid class IDs found in y.")

class_counts = np.bincount(y, minlength=NUM_CLASSES)

print("\nClass distribution:")
print("Minimum samples/class:", class_counts.min())
print("Maximum samples/class:", class_counts.max())
print("Average samples/class:", round(class_counts.mean(), 2))


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 75)
print("TRAIN / VALIDATION SPLIT")
print("=" * 75)

# IMPORTANT:
# There is NO test split from X.npy.
# Every augmented landmark sample is used either for
# training or validation.

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("External test: separate image folder")


# ============================================================
# STANDARD SCALER
# ============================================================

print("\n" + "=" * 75)
print("FEATURE STANDARDIZATION")
print("=" * 75)

scaler = StandardScaler()

# Fit ONLY on training data.
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

SCALER_PATH = MODEL_DIR / "scaler.pkl"

with open(SCALER_PATH, "wb") as file:
    pickle.dump(scaler, file)

print("Scaler saved:", SCALER_PATH)


# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 75)
print("BUILDING MODEL")
print("=" * 75)

model = Sequential([
    Input(shape=(NUM_FEATURES,), name="landmark_input"),

    Dense(352, activation="relu", name="dense_352"),
    BatchNormalization(name="batch_norm_352"),
    Dropout(DROPOUT_1, name="dropout_352"),

    Dense(256, activation="relu", name="dense_256"),
    BatchNormalization(name="batch_norm_256"),
    Dropout(DROPOUT_2, name="dropout_256"),

    Dense(128, activation="relu", name="dense_128"),
    BatchNormalization(name="batch_norm_128"),
    Dropout(DROPOUT_3, name="dropout_128"),

    Dense(64, activation="relu", name="dense_64"),
    Dropout(DROPOUT_4, name="dropout_64"),

    Dense(NUM_CLASSES, activation="softmax", name="output")
])

model.summary()

print("\nTotal parameters:", model.count_params())


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

BEST_MODEL_PATH = MODEL_DIR / "isl_229_best_external_test.keras"

callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),

    ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
]


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 75)
print("STARTING TRAINING")
print("=" * 75)

print("Training device:", "NVIDIA GPU" if gpus else "CPU")
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print(
    "Dropout:",
    DROPOUT_1,
    DROPOUT_2,
    DROPOUT_3,
    DROPOUT_4
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# LOAD BEST VALIDATION MODEL
# ============================================================

print("\nLoading best validation model...")
model = tf.keras.models.load_model(BEST_MODEL_PATH)


# ============================================================
# MEDIAPIPE EXTERNAL TEST FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 75)
print("PREPARING EXTERNAL TEST DATA")
print("=" * 75)

if not TEST_IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"External test directory not found:\n{TEST_IMAGE_DIR}"
    )


def normalize_hand(hand_landmarks, left_shoulder, right_shoulder):
    """Return 63 normalized features for one hand."""

    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    shoulder_width = np.linalg.norm(
        left_shoulder[:2] - right_shoulder[:2]
    )

    if shoulder_width < 1e-6:
        shoulder_width = 1.0

    features = []

    for landmark in hand_landmarks.landmark:
        point = np.array(
            [landmark.x, landmark.y, landmark.z],
            dtype=np.float32
        )

        point = (point - shoulder_center) / shoulder_width
        features.extend(point.tolist())

    return np.asarray(features, dtype=np.float32)


def extract_135_features(image_path, hands_detector, pose_detector):
    """
    135 features:
        Left hand       = 63
        Right hand      = 63
        Left shoulder   = 3
        Right shoulder  = 3
        Presence flags  = 3
        Total           = 135
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return None, "image_read_failed"

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------
    # POSE / SHOULDERS
    # --------------------------------------------------------

    pose_result = pose_detector.process(rgb)

    left_shoulder = DEFAULT_LEFT_SHOULDER.copy()
    right_shoulder = DEFAULT_RIGHT_SHOULDER.copy()
    shoulder_present = 0.0

    if pose_result.pose_landmarks is not None:

        landmarks = pose_result.pose_landmarks.landmark

        left = landmarks[11]
        right = landmarks[12]

        if (
            left.visibility >= POSE_VISIBILITY_THRESHOLD
            and right.visibility >= POSE_VISIBILITY_THRESHOLD
        ):
            left_shoulder = np.array(
                [left.x, left.y, left.z],
                dtype=np.float32
            )

            right_shoulder = np.array(
                [right.x, right.y, right.z],
                dtype=np.float32
            )

            shoulder_present = 1.0

    # --------------------------------------------------------
    # HANDS
    # --------------------------------------------------------

    hand_result = hands_detector.process(rgb)

    left_hand_features = np.zeros(63, dtype=np.float32)
    right_hand_features = np.zeros(63, dtype=np.float32)

    left_present = 0.0
    right_present = 0.0

    if hand_result.multi_hand_landmarks is not None:

        handedness_list = hand_result.multi_handedness

        for hand_landmarks, handedness in zip(
            hand_result.multi_hand_landmarks,
            handedness_list
        ):

            label = handedness.classification[0].label.lower()

            hand_features = normalize_hand(
                hand_landmarks,
                left_shoulder,
                right_shoulder
            )

            if label == "left":
                left_hand_features = hand_features
                left_present = 1.0

            elif label == "right":
                right_hand_features = hand_features
                right_present = 1.0

    features = np.concatenate([
        left_hand_features,
        right_hand_features,
        left_shoulder,
        right_shoulder,
        np.array(
            [
                left_present,
                right_present,
                shoulder_present
            ],
            dtype=np.float32
        )
    ])

    if features.shape[0] != 135:
        raise ValueError(
            f"Expected 135 features, got {features.shape[0]}"
        )

    return features, "ok"


# ============================================================
# READ TEST FOLDERS
# ============================================================

class_names = [str(label) for label in classes]
class_to_id = {
    name: idx
    for idx, name in enumerate(class_names)
}

test_features = []
test_labels = []
test_paths = []

unknown_classes = []
empty_classes = []
failed_images = []

class_folders = sorted(
    [
        p for p in TEST_IMAGE_DIR.iterdir()
        if p.is_dir()
    ],
    key=lambda p: p.name.lower()
)

print("Test class folders found:", len(class_folders))


# ============================================================
# EXTRACT TEST FEATURES
# ============================================================

with mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=HAND_DETECTION_CONFIDENCE
) as hands_detector, \
    mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=POSE_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_TRACKING_CONFIDENCE
    ) as pose_detector:

    for class_folder in class_folders:

        class_name = class_folder.name

        # Test set can contain fewer than all 229 classes.
        if class_name not in class_to_id:
            unknown_classes.append(class_name)
            continue

        image_paths = sorted(
            [
                p for p in class_folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda p: p.name.lower()
        )

        if not image_paths:
            empty_classes.append(class_name)
            continue

        label_id = class_to_id[class_name]

        for image_path in image_paths:

            try:
                features, status = extract_135_features(
                    image_path,
                    hands_detector,
                    pose_detector
                )

                if features is None:
                    failed_images.append(
                        (str(image_path), status)
                    )
                    continue

                test_features.append(features)
                test_labels.append(label_id)
                test_paths.append(str(image_path))

            except Exception as exc:
                failed_images.append(
                    (str(image_path), str(exc))
                )


if not test_features:
    raise RuntimeError(
        "No valid external test images were extracted."
    )

X_external_test = np.asarray(
    test_features,
    dtype=np.float32
)

y_external_test = np.asarray(
    test_labels,
    dtype=np.int32
)

print("\nExternal test data:")
print(
    "Images successfully extracted:",
    len(X_external_test)
)
print(
    "Features:",
    X_external_test.shape[1]
)
print(
    "Classes represented:",
    len(np.unique(y_external_test))
)

if unknown_classes:
    print(
        "\nWARNING - test classes NOT found "
        "in training classes:"
    )
    for name in unknown_classes:
        print("  ", name)

if empty_classes:
    print("\nWARNING - empty test folders:")
    for name in empty_classes:
        print("  ", name)

if failed_images:
    print(
        "\nWARNING - failed image extractions:",
        len(failed_images)
    )


# ============================================================
# SCALE EXTERNAL TEST DATA
# ============================================================

# IMPORTANT:
# Use the scaler fitted on X_train.
X_external_test_scaled = scaler.transform(
    X_external_test
)


# ============================================================
# FINAL EXTERNAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL EXTERNAL TEST EVALUATION")
print("=" * 75)

test_loss, test_accuracy = model.evaluate(
    X_external_test_scaled,
    y_external_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

print("\nExternal test loss:", round(test_loss, 5))

print(
    "EXTERNAL TEST ACCURACY:",
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating external test predictions...")

probabilities = model.predict(
    X_external_test_scaled,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)
print("EXTERNAL TEST CLASSIFICATION REPORT")
print("=" * 75)

present_labels = np.unique(
    np.concatenate([
        y_external_test,
        y_pred
    ])
)

report = classification_report(
    y_external_test,
    y_pred,
    labels=present_labels,
    target_names=[
        class_names[i]
        for i in present_labels
    ],
    zero_division=0
)

print(report)

REPORT_PATH = (
    MODEL_DIR /
    "external_test_classification_report.txt"
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_external_test,
    y_pred,
    labels=present_labels
)

CM_PATH = (
    MODEL_DIR /
    "external_test_confusion_matrix.npy"
)

np.save(CM_PATH, cm)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_PATH = (
    MODEL_DIR /
    "isl_229_final_external_test.keras"
)

model.save(FINAL_MODEL_PATH)


# ============================================================
# SAVE CLASSES
# ============================================================

CLASSES_JSON_PATH = (
    MODEL_DIR /
    "classes.json"
)

with open(
    CLASSES_JSON_PATH,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        class_names,
        file,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

HISTORY_PATH = (
    MODEL_DIR /
    "history_external_test.json"
)

history_dict = {
    key: [
        float(value)
        for value in values
    ]
    for key, values in history.history.items()
}

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        history_dict,
        file,
        indent=2
    )


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

MODEL_INFO_PATH = (
    MODEL_DIR /
    "model_info_external_test.json"
)

model_info = {
    "tensorflow_version": tf.__version__,
    "gpu_detected": len(gpus) > 0,
    "gpu": (
        str(gpus[0])
        if gpus
        else "CPU"
    ),

    "number_of_classes": int(NUM_CLASSES),
    "number_of_features": int(NUM_FEATURES),

    "total_augmented_samples": int(NUM_SAMPLES),
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),

    "external_test_images": int(
        len(X_external_test)
    ),
    "external_test_classes": int(
        len(np.unique(y_external_test))
    ),

    "architecture": [
        int(NUM_FEATURES),
        352,
        256,
        128,
        64,
        int(NUM_CLASSES)
    ],

    "dropout_rates": [
        DROPOUT_1,
        DROPOUT_2,
        DROPOUT_3,
        DROPOUT_4
    ],

    "optimizer": "Adam",
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "maximum_epochs": EPOCHS,

    "total_parameters": int(
        model.count_params()
    ),

    "external_test_loss": float(
        test_loss
    ),
    "external_test_accuracy": float(
        test_accuracy
    )
}

with open(
    MODEL_INFO_PATH,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        model_info,
        file,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("TRAINING + EXTERNAL TEST COMPLETE")
print("=" * 75)

print("\nClasses:", NUM_CLASSES)
print("Features:", NUM_FEATURES)
print(
    "Total parameters:",
    model.count_params()
)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)

print(
    "External test images:",
    len(X_external_test)
)

print("\nFINAL EXTERNAL TEST ACCURACY:")
print(
    f"{test_accuracy * 100:.2f}%"
)

print("\nSaved files:")
print("1. Final model:", FINAL_MODEL_PATH)
print("2. Best model:", BEST_MODEL_PATH)
print("3. Scaler:", SCALER_PATH)
print("4. Classes:", CLASSES_JSON_PATH)
print("5. Classification report:", REPORT_PATH)
print("6. Confusion matrix:", CM_PATH)
print("7. Training history:", HISTORY_PATH)
print("8. Model information:", MODEL_INFO_PATH)

print("\nDONE")
