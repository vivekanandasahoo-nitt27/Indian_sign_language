from pathlib import Path
import json
import pickle
import time

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = Path(r"D:\aug_indian_dataset")

# Landmark output
OUTPUT_DIR = Path(r"D:\indian_sign_landmarks")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# MediaPipe confidence
HAND_MIN_DETECTION = 0.50
POSE_MIN_DETECTION = 0.20
POSE_VISIBILITY_THRESHOLD = 0.20

# ------------------------------------------------------------
# DEFAULT SHOULDER REFERENCE
#
# These are the real normalized MediaPipe shoulder coordinates
# you just obtained from D:\baby.jpg.
#
# They are used ONLY when both real shoulders are unavailable.
# ------------------------------------------------------------

DEFAULT_LEFT_SHOULDER = np.array(
    [0.69371974, 0.36440188, -0.13180971],
    dtype=np.float32,
)

DEFAULT_RIGHT_SHOULDER = np.array(
    [0.31984484, 0.37018980, -0.17071517],
    dtype=np.float32,
)


# ============================================================
# 231-CLASS DATASET
# ============================================================

# The folders themselves are the labels.
# Sorting gives one fixed class index mapping for training.
#
# Expected:
# 195 ISL classes + 9 numbers + 26 letters = 230
#
# NOTE:
# 9 + 26 = 35, not 36.
# If your dataset contains another class, the script will simply
# use whatever folders actually exist and report the count.
# ============================================================


def get_class_folders():
    folders = sorted(
        [
            p
            for p in DATASET_DIR.iterdir()
            if p.is_dir()
        ],
        key=lambda p: p.name.lower(),
    )

    return folders


def get_images(folder):
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )


# ============================================================
# SHOULDER FUNCTIONS
# ============================================================

def get_shoulders(pose_result):
    """
    Return:
        shoulders: shape (2, 3)
        shoulder_present: 1 if both real shoulders are usable,
                          otherwise 0

    If both real shoulders are visible, use them.

    Otherwise use the fixed default shoulder reference supplied
    from the user's shoulder test.
    """

    if pose_result.pose_landmarks is not None:

        landmarks = pose_result.pose_landmarks.landmark

        left = landmarks[
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER
        ]

        right = landmarks[
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
        ]

        if (
            left.visibility >= POSE_VISIBILITY_THRESHOLD
            and
            right.visibility >= POSE_VISIBILITY_THRESHOLD
        ):

            shoulders = np.array(
                [
                    [left.x, left.y, left.z],
                    [right.x, right.y, right.z],
                ],
                dtype=np.float32,
            )

            return shoulders, 1


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    shoulders = np.array(
        [
            DEFAULT_LEFT_SHOULDER,
            DEFAULT_RIGHT_SHOULDER,
        ],
        dtype=np.float32,
    )

    return shoulders, 0


# ============================================================
# HAND FEATURE EXTRACTION
# ============================================================

def get_hand_landmarks(hand_result):
    """
    Return:
        left_hand  = 21x3
        right_hand = 21x3
        left_present
        right_present

    MediaPipe is run on the original, non-mirrored image.
    """

    left_hand = np.zeros(
        (21, 3),
        dtype=np.float32,
    )

    right_hand = np.zeros(
        (21, 3),
        dtype=np.float32,
    )

    left_present = 0
    right_present = 0


    if hand_result.multi_hand_landmarks is None:
        return (
            left_hand,
            right_hand,
            left_present,
            right_present,
        )


    for hand_landmarks, handedness in zip(
        hand_result.multi_hand_landmarks,
        hand_result.multi_handedness,
    ):

        points = np.array(
            [
                [
                    lm.x,
                    lm.y,
                    lm.z,
                ]
                for lm in hand_landmarks.landmark
            ],
            dtype=np.float32,
        )


        label = handedness.classification[0].label


        if label == "Left":

            left_hand = points
            left_present = 1

        elif label == "Right":

            right_hand = points
            right_present = 1


    return (
        left_hand,
        right_hand,
        left_present,
        right_present,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_hand(
    hand,
    shoulders,
):
    """
    Normalize hand coordinates relative to the shoulder frame.

    1. Find shoulder center.
    2. Calculate shoulder width.
    3. Subtract shoulder center.
    4. Divide by shoulder width.

    This makes the hand representation much less dependent
    on camera distance and image position.
    """

    shoulder_center = (
        shoulders[0] +
        shoulders[1]
    ) / 2.0


    shoulder_width = np.linalg.norm(
        shoulders[0][:2] -
        shoulders[1][:2]
    )


    # Safety against degenerate reference
    if shoulder_width < 1e-6:
        shoulder_width = 0.374


    normalized = (
        hand -
        shoulder_center
    ) / shoulder_width


    return normalized.astype(
        np.float32
    )


# ============================================================
# FEATURE VECTOR
# ============================================================

def extract_features(
    hand_result,
    pose_result,
):
    """
    Final feature layout:

        Left hand       21*3 = 63
        Right hand      21*3 = 63
        Left shoulder          3
        Right shoulder         3
        Left present           1
        Right present          1
        Shoulder present       1
        --------------------------------
        TOTAL                 135
    """

    (
        left_hand,
        right_hand,
        left_present,
        right_present,
    ) = get_hand_landmarks(
        hand_result
    )


    shoulders, shoulder_present = (
        get_shoulders(
            pose_result
        )
    )


    # Normalize only real hand coordinates.
    # Missing hand remains zeros.
    if left_present:
        left_normalized = normalize_hand(
            left_hand,
            shoulders,
        )
    else:
        left_normalized = np.zeros(
            (21, 3),
            dtype=np.float32,
        )


    if right_present:
        right_normalized = normalize_hand(
            right_hand,
            shoulders,
        )
    else:
        right_normalized = np.zeros(
            (21, 3),
            dtype=np.float32,
        )


    features = np.concatenate(
        [
            left_normalized.reshape(-1),
            right_normalized.reshape(-1),

            # Keep shoulder information in the
            # same normalized MediaPipe coordinate
            # system used by the images.
            shoulders[0],
            shoulders[1],

            np.array(
                [
                    left_present,
                    right_present,
                    shoulder_present,
                ],
                dtype=np.float32,
            ),
        ]
    )


    # Must always be 135
    if features.shape[0] != 135:

        raise RuntimeError(
            f"Expected 135 features, "
            f"got {features.shape[0]}"
        )


    return features.astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)
    print("INDIAN SIGN LANGUAGE - LANDMARK EXTRACTION")
    print("=" * 80)

    print("\nDataset:")
    print(DATASET_DIR)

    print("\nOutput:")
    print(OUTPUT_DIR)

    print("\nFeature size: 135")

    print(
        "\nDefault shoulder fallback:"
    )

    print(
        "Left :",
        DEFAULT_LEFT_SHOULDER
    )

    print(
        "Right:",
        DEFAULT_RIGHT_SHOULDER
    )


    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_DIR}"
        )


    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    class_folders = get_class_folders()

    if not class_folders:

        raise RuntimeError(
            "No class folders found."
        )


    print(
        f"\nClasses found: "
        f"{len(class_folders)}"
    )


    # --------------------------------------------------------
    # Save class mapping
    # --------------------------------------------------------

    class_names = [
        folder.name
        for folder in class_folders
    ]


    np.save(
        OUTPUT_DIR / "classes.npy",
        np.array(
            class_names,
            dtype=object,
        ),
    )


    with open(
        OUTPUT_DIR / "classes.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            class_names,
            f,
            indent=2,
            ensure_ascii=False,
        )


    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose


    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=HAND_MIN_DETECTION,
    )


    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=POSE_MIN_DETECTION,
        min_tracking_confidence=POSE_MIN_DETECTION,
    )


    # --------------------------------------------------------
    # Output arrays
    # --------------------------------------------------------

    all_features = []
    all_labels = []
    all_paths = []


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_images = 0
    processed = 0
    failed = 0
    hand_detected = 0
    both_hands = 0
    pose_detected = 0
    default_shoulder_used = 0


    start_time = time.time()


    # ========================================================
    # PROCESS CLASSES
    # ========================================================

    for class_index, class_folder in enumerate(
        class_folders
    ):

        images = get_images(
            class_folder
        )


        print(
            f"\n[{class_index + 1:>3}/"
            f"{len(class_folders):<3}] "
            f"{class_folder.name}"
        )

        print(
            f"    Images: {len(images)}"
        )


        for image_number, image_path in enumerate(
            images,
            start=1,
        ):

            total_images += 1


            image = cv2.imread(
                str(image_path)
            )


            if image is None:

                failed += 1

                continue


            try:

                rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                )


                # --------------------------------------------
                # HANDS
                # --------------------------------------------

                hand_result = hands.process(
                    rgb
                )


                # --------------------------------------------
                # POSE
                # --------------------------------------------

                pose_result = pose.process(
                    rgb
                )


                # --------------------------------------------
                # Statistics
                # --------------------------------------------

                hand_count = 0

                if hand_result.multi_hand_landmarks:

                    hand_count = len(
                        hand_result.multi_hand_landmarks
                    )


                if hand_count >= 1:
                    hand_detected += 1

                if hand_count >= 2:
                    both_hands += 1


                if pose_result.pose_landmarks:

                    landmarks = (
                        pose_result.pose_landmarks.landmark
                    )

                    left = landmarks[
                        mp_pose.PoseLandmark.LEFT_SHOULDER
                    ]

                    right = landmarks[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER
                    ]

                    if (
                        left.visibility
                        >= POSE_VISIBILITY_THRESHOLD
                        and
                        right.visibility
                        >= POSE_VISIBILITY_THRESHOLD
                    ):

                        pose_detected += 1


                # --------------------------------------------
                # FEATURES
                # --------------------------------------------

                features = extract_features(
                    hand_result,
                    pose_result,
                )


                # Check whether fallback was used
                _, shoulder_flag = get_shoulders(
                    pose_result
                )

                if shoulder_flag == 0:
                    default_shoulder_used += 1


                all_features.append(
                    features
                )

                all_labels.append(
                    class_index
                )

                # Store relative dataset path
                all_paths.append(
                    str(
                        image_path.relative_to(
                            DATASET_DIR
                        )
                    )
                )


                processed += 1


            except Exception as error:

                failed += 1

                print(
                    f"\n    ERROR: "
                    f"{image_path}"
                )

                print(
                    f"    {error}"
                )


            # Progress every 100 images
            if (
                image_number % 100 == 0
                or
                image_number == len(images)
            ):

                print(
                    f"\r    Progress: "
                    f"{image_number}/"
                    f"{len(images)}",
                    end="",
                )


        print()


    # ========================================================
    # CLOSE MEDIAPIPE
    # ========================================================

    hands.close()
    pose.close()


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    if not all_features:

        raise RuntimeError(
            "No features were extracted."
        )


    X = np.asarray(
        all_features,
        dtype=np.float32,
    )

    y = np.asarray(
        all_labels,
        dtype=np.int32,
    )


    # ========================================================
    # SAVE
    # ========================================================

    np.save(
        OUTPUT_DIR / "X.npy",
        X,
    )

    np.save(
        OUTPUT_DIR / "y.npy",
        y,
    )

    np.save(
        OUTPUT_DIR / "image_paths.npy",
        np.array(
            all_paths,
            dtype=object,
        ),
    )


    # Metadata
    metadata = {
        "dataset": str(DATASET_DIR),
        "classes": len(class_names),
        "samples": int(len(X)),
        "features": int(X.shape[1]),
        "hand_features": 126,
        "shoulder_features": 6,
        "presence_features": 3,
        "hand_detection_min_confidence": HAND_MIN_DETECTION,
        "pose_detection_min_confidence": POSE_MIN_DETECTION,
        "pose_visibility_threshold": POSE_VISIBILITY_THRESHOLD,
        "default_left_shoulder": DEFAULT_LEFT_SHOULDER.tolist(),
        "default_right_shoulder": DEFAULT_RIGHT_SHOULDER.tolist(),
        "hand_detected": int(hand_detected),
        "both_hands": int(both_hands),
        "pose_detected": int(pose_detected),
        "default_shoulder_used": int(default_shoulder_used),
        "failed": int(failed),
    }


    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


    # ========================================================
    # CLASS COUNTS
    # ========================================================

    counts = np.bincount(
        y,
        minlength=len(class_names),
    )


    with open(
        OUTPUT_DIR / "class_counts.txt",
        "w",
        encoding="utf-8",
    ) as f:

        for index, name in enumerate(
            class_names
        ):

            f.write(
                f"{index:03d} "
                f"{name:<35} "
                f"{counts[index]}\n"
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    elapsed = (
        time.time() - start_time
    )


    print("\n" + "=" * 80)
    print("LANDMARK EXTRACTION COMPLETE")
    print("=" * 80)

    print(
        f"Classes              : "
        f"{len(class_names):,}"
    )

    print(
        f"Images found         : "
        f"{total_images:,}"
    )

    print(
        f"Successfully processed: "
        f"{processed:,}"
    )

    print(
        f"Failed               : "
        f"{failed:,}"
    )

    print(
        f"Hand detected        : "
        f"{hand_detected:,}"
    )

    print(
        f"Both hands detected  : "
        f"{both_hands:,}"
    )

    print(
        f"Real pose detected   : "
        f"{pose_detected:,}"
    )

    print(
        f"Default shoulders    : "
        f"{default_shoulder_used:,}"
    )

    print(
        f"Feature shape        : "
        f"{X.shape}"
    )

    print(
        f"Time taken           : "
        f"{elapsed / 60:.2f} minutes"
    )

    print("\nSaved files:")

    print(
        OUTPUT_DIR / "X.npy"
    )

    print(
        OUTPUT_DIR / "y.npy"
    )

    print(
        OUTPUT_DIR / "classes.npy"
    )

    print(
        OUTPUT_DIR / "classes.json"
    )

    print(
        OUTPUT_DIR / "image_paths.npy"
    )

    print(
        OUTPUT_DIR / "metadata.json"
    )

    print(
        OUTPUT_DIR / "class_counts.txt"
    )

    print("\nFeature layout:")
    print("  Left hand       : 63")
    print("  Right hand      : 63")
    print("  Left shoulder   : 3")
    print("  Right shoulder  : 3")
    print("  Presence flags  : 3")
    print("  TOTAL           : 135")

    print("=" * 80)


if __name__ == "__main__":
    main()
