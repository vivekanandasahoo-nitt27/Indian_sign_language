import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


# ============================================================
# IMAGE
# ============================================================

IMAGE_PATH = Path(r"D:\baby.jpg")


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {IMAGE_PATH}"
    )

height, width = image.shape[:2]

print("=" * 70)
print("SHOULDER TEST")
print("=" * 70)

print("Image:", IMAGE_PATH)
print("Size:", width, "x", height)


rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)


pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.20,
    min_tracking_confidence=0.20
)


# ============================================================
# DETECTION
# ============================================================

print("\nRunning MediaPipe...")

hand_result = hands.process(rgb)
pose_result = pose.process(rgb)


# ============================================================
# HANDS
# ============================================================

wrist_points = []

if hand_result.multi_hand_landmarks:

    print(
        "Hands detected:",
        len(hand_result.multi_hand_landmarks)
    )

    for hand in hand_result.multi_hand_landmarks:

        wrist = hand.landmark[
            mp_hands.HandLandmark.WRIST
        ]

        wrist_points.append(
            np.array(
                [wrist.x, wrist.y, wrist.z],
                dtype=np.float32
            )
        )

else:

    print("Hands detected: 0")


# ============================================================
# SHOULDERS
# ============================================================

real_shoulders = None

if pose_result.pose_landmarks:

    landmarks = pose_result.pose_landmarks.landmark

    left = landmarks[
        mp_pose.PoseLandmark.LEFT_SHOULDER
    ]

    right = landmarks[
        mp_pose.PoseLandmark.RIGHT_SHOULDER
    ]

    print("\nPose detected")

    print(
        "Left shoulder visibility:",
        round(left.visibility, 3)
    )

    print(
        "Right shoulder visibility:",
        round(right.visibility, 3)
    )


    if (
        left.visibility >= 0.20
        and
        right.visibility >= 0.20
    ):

        real_shoulders = np.array([
            [left.x, left.y, left.z],
            [right.x, right.y, right.z]
        ], dtype=np.float32)


# ============================================================
# DEFAULT SHOULDERS
# ============================================================

def create_default_shoulders(wrists):

    print(
        "\nCreating DEFAULT shoulders..."
    )

    # Two hands
    if len(wrists) >= 2:

        w1 = wrists[0]
        w2 = wrists[1]

        center = (w1 + w2) / 2.0

        distance = np.linalg.norm(
            w1[:2] - w2[:2]
        )

        distance = max(distance, 0.10)

        shoulder_width = distance * 2.0

        vertical_offset = distance * 0.60

        left = center.copy()
        right = center.copy()

        left[0] -= shoulder_width / 2
        right[0] += shoulder_width / 2

        left[1] -= vertical_offset
        right[1] -= vertical_offset

        return np.array(
            [left, right],
            dtype=np.float32
        )


    # One hand
    elif len(wrists) == 1:

        wrist = wrists[0]

        left = wrist.copy()
        right = wrist.copy()

        left[0] -= 0.175
        right[0] += 0.175

        left[1] -= 0.20
        right[1] -= 0.20

        return np.array(
            [left, right],
            dtype=np.float32
        )


    # No hands
    else:

        return np.array([
            [-0.25, 0.35, 0.0],
            [ 0.25, 0.35, 0.0]
        ], dtype=np.float32)


# ============================================================
# SELECT REAL OR DEFAULT
# ============================================================

if real_shoulders is not None:

    shoulders = real_shoulders
    shoulder_present = 1

    print("\nREAL SHOULDERS DETECTED")

else:

    shoulders = create_default_shoulders(
        wrist_points
    )

    shoulder_present = 0

    print(
        "\nREAL SHOULDERS NOT DETECTED"
    )

    print(
        "Using DEFAULT shoulders"
    )


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    "Left shoulder:",
    shoulders[0]
)

print(
    "Right shoulder:",
    shoulders[1]
)

print(
    "Shoulder present:",
    shoulder_present
)


# ============================================================
# DRAW TEST RESULT
# ============================================================

output = image.copy()


for point, name in zip(
    shoulders,
    ["LEFT", "RIGHT"]
):

    x = int(point[0] * width)
    y = int(point[1] * height)

    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    cv2.circle(
        output,
        (x, y),
        12,
        (0, 255, 0),
        -1
    )

    cv2.putText(
        output,
        name,
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


# Draw wrists

for wrist in wrist_points:

    x = int(wrist[0] * width)
    y = int(wrist[1] * height)

    cv2.circle(
        output,
        (x, y),
        8,
        (255, 0, 0),
        -1
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH = Path(
    r"D:\shoulder_test_result.jpg"
)

cv2.imwrite(
    str(OUTPUT_PATH),
    output
)

print("\nResult saved to:")
print(OUTPUT_PATH)


# ============================================================
# CLEANUP
# ============================================================

hands.close()
pose.close()