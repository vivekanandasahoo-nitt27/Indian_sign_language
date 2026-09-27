import base64
import json
import pickle
import time
from collections import deque
from pathlib import Path
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "indian_sign_model"

MODEL_PATH = MODEL_DIR / "isl_229_best_external_test.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
CLASSES_PATH = MODEL_DIR / "classes.json"

CONFIDENCE_THRESHOLD = 0.70
REQUIRED_CONSECUTIVE = 5
HISTORY_SECONDS = 20.0

HAND_CONFIDENCE = 0.50
POSE_CONFIDENCE = 0.20
POSE_VISIBILITY = 0.20

DEFAULT_LEFT_SHOULDER = np.array(
    [0.69371974, 0.36440188, -0.13180971], dtype=np.float32
)
DEFAULT_RIGHT_SHOULDER = np.array(
    [0.31984484, 0.37018980, -0.17071517], dtype=np.float32
)

app = Flask(__name__)

for path in (MODEL_PATH, SCALER_PATH, CLASSES_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading scaler...")
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("Loading classes...")
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = [str(x) for x in json.load(f)]

if model.input_shape[-1] != 135:
    raise ValueError(f"Expected 135 model inputs, got {model.input_shape[-1]}")
if model.output_shape[-1] != len(classes):
    raise ValueError(
        f"Model outputs={model.output_shape[-1]}, classes={len(classes)}"
    )

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=HAND_CONFIDENCE,
)

pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=POSE_CONFIDENCE,
    min_tracking_confidence=POSE_CONFIDENCE,
)

state_lock = Lock()

history = deque()  # (timestamp, word)
last_prediction = None
consecutive_count = 0
confirmed_sign = None
release_count = 0
last_confidence = 0.0
last_inference_ms = 0.0


def prune_history(now=None):
    now = time.time() if now is None else now
    while history and now - history[0][0] > HISTORY_SECONDS:
        history.popleft()


def normalize_hand(hand_landmarks, left_shoulder, right_shoulder):
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
            dtype=np.float32,
        )
        point = (point - shoulder_center) / shoulder_width
        features.extend(point.tolist())

    return np.asarray(features, dtype=np.float32)


def extract_features(rgb_image):
    """Same 135-feature order as the training pipeline."""
    pose_result = pose_detector.process(rgb_image)

    left_shoulder = DEFAULT_LEFT_SHOULDER.copy()
    right_shoulder = DEFAULT_RIGHT_SHOULDER.copy()
    shoulder_present = 0.0

    if pose_result.pose_landmarks is not None:
        landmarks = pose_result.pose_landmarks.landmark
        left = landmarks[11]
        right = landmarks[12]

        if (
            left.visibility >= POSE_VISIBILITY
            and right.visibility >= POSE_VISIBILITY
        ):
            left_shoulder = np.array(
                [left.x, left.y, left.z], dtype=np.float32
            )
            right_shoulder = np.array(
                [right.x, right.y, right.z], dtype=np.float32
            )
            shoulder_present = 1.0

    hand_result = hands_detector.process(rgb_image)

    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)
    left_present = 0.0
    right_present = 0.0

    if hand_result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
            hand_result.multi_hand_landmarks,
            hand_result.multi_handedness,
        ):
            label = handedness.classification[0].label.lower()
            hand_features = normalize_hand(
                hand_landmarks, left_shoulder, right_shoulder
            )

            if label == "left":
                left_hand = hand_features
                left_present = 1.0
            elif label == "right":
                right_hand = hand_features
                right_present = 1.0

    features = np.concatenate(
        [
            left_hand,
            right_hand,
            left_shoulder,
            right_shoulder,
            np.array(
                [left_present, right_present, shoulder_present],
                dtype=np.float32,
            ),
        ]
    )

    if features.shape != (135,):
        raise ValueError(f"Expected 135 features, got {features.shape}")

    return features


def decode_frame(data_url):
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("Invalid frame data")

    _, encoded = data_url.split(",", 1)
    raw = base64.b64decode(encoded)
    image_array = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError("Could not decode image")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def predict_frame(rgb_image):
    global last_confidence, last_inference_ms

    start = time.perf_counter()

    features = extract_features(rgb_image)
    scaled = scaler.transform(features.reshape(1, -1)).astype(np.float32)

    probabilities = model.predict(scaled, verbose=0)[0]
    index = int(np.argmax(probabilities))

    label = classes[index]
    confidence = float(probabilities[index])

    last_confidence = confidence
    last_inference_ms = (time.perf_counter() - start) * 1000.0

    return label, confidence


def update_temporal_state(label, confidence):
    """
    Confirm a sign after 5 consecutive sampled frames.

    After confirmation, the same held sign is locked so it is not
    appended repeatedly. A different/low-confidence sequence releases it.
    """
    global last_prediction, consecutive_count
    global confirmed_sign, release_count

    now = time.time()
    prune_history(now)

    if confidence < CONFIDENCE_THRESHOLD:
        last_prediction = None
        consecutive_count = 0

        if confirmed_sign is not None:
            release_count += 1
            if release_count >= REQUIRED_CONSECUTIVE:
                confirmed_sign = None
                release_count = 0
        return None

    if label == last_prediction:
        consecutive_count += 1
    else:
        last_prediction = label
        consecutive_count = 1

    if confirmed_sign == label:
        release_count = 0
        return None

    if confirmed_sign is not None and label != confirmed_sign:
        release_count += 1
        if release_count < REQUIRED_CONSECUTIVE:
            return None
        confirmed_sign = None
        release_count = 0

    if consecutive_count >= REQUIRED_CONSECUTIVE:
        confirmed_sign = label
        consecutive_count = 0

        if not history or history[-1][1] != label:
            history.append((now, label))
            prune_history(now)

        return label

    return None


def make_response(prediction=None, confirmed=None):
    now = time.time()
    prune_history(now)

    return {
        "prediction": prediction,
        "confidence": round(last_confidence, 4),
        "confirmed": confirmed,
        "current_sign": confirmed_sign,
        "sentence": " ".join(word for _, word in history),
        "history": [
            {"word": word, "age": round(now - ts, 2)}
            for ts, word in history
        ],
        "consecutive_count": consecutive_count,
        "required_consecutive": REQUIRED_CONSECUTIVE,
        "inference_ms": round(last_inference_ms, 2),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    with state_lock:
        return jsonify(make_response())


@app.post("/api/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        frame = payload.get("frame")
        if not frame:
            return jsonify({"error": "Missing frame"}), 400

        rgb_image = decode_frame(frame)

        with state_lock:
            label, confidence = predict_frame(rgb_image)
            confirmed = update_temporal_state(label, confidence)
            return jsonify(make_response(label, confirmed))

    except Exception as exc:
        app.logger.exception("Prediction error")
        return jsonify({"error": str(exc)}), 500


@app.post("/api/reset")
def reset():
    global last_prediction, consecutive_count
    global confirmed_sign, release_count, last_confidence

    with state_lock:
        history.clear()
        last_prediction = None
        consecutive_count = 0
        confirmed_sign = None
        release_count = 0
        last_confidence = 0.0

    return jsonify({"ok": True})


if __name__ == "__main__":
    print("=" * 70)
    print("INDIAN SIGN LANGUAGE LIVE TRANSLATOR")
    print("=" * 70)
    print(f"Model      : {MODEL_PATH}")
    print(f"Classes    : {len(classes)}")
    print("Features   : 135")
    print(f"Parameters : {model.count_params()}")
    print(f"Threshold  : {CONFIDENCE_THRESHOLD}")
    print(f"Confirm    : {REQUIRED_CONSECUTIVE} frames")
    print(f"History    : {HISTORY_SECONDS} seconds")
    print("URL        : http://127.0.0.1:5000")
    print("=" * 70)

    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
