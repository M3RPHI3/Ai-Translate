from flask import Flask, render_template, jsonify
import threading
import collections
import time

import cv2
import mediapipe as mp
import tst

app = Flask(__name__)

latest_word = "Waiting..."

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

TARGET_W, TARGET_H = 1280, 720

CONFIDENCE_THRESHOLD = 0.70
STABILITY_FRAMES = 10

gesture_buffer = collections.deque(maxlen=STABILITY_FRAMES)

# ──────────────────────────────────────────────
# MediaPipe
# ──────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ──────────────────────────────────────────────
# Results Wrapper
# ──────────────────────────────────────────────

class _Results:

    def __init__(self, hands_res, pose_res):

        self.pose_landmarks = (
            pose_res.pose_landmarks
            if pose_res else None
        )

        self.left_hand_landmarks = None
        self.right_hand_landmarks = None

        if hands_res and hands_res.multi_hand_landmarks:

            for lms, handed in zip(
                hands_res.multi_hand_landmarks,
                hands_res.multi_handedness
            ):

                mp_label = handed.classification[0].label

                if mp_label == "Left":
                    self.right_hand_landmarks = lms
                else:
                    self.left_hand_landmarks = lms

# ──────────────────────────────────────────────
# Camera Loop
# ──────────────────────────────────────────────

def camera_loop():

    global latest_word

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(
            frame,
            (TARGET_W, TARGET_H)
        )

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False

        hands_res = hands_detector.process(rgb)
        pose_res = pose_detector.process(rgb)

        rgb.flags.writeable = True

        results = _Results(
            hands_res,
            pose_res
        )

        has_hand = (
            results.left_hand_landmarks
            or results.right_hand_landmarks
        )

        if not has_hand:
            latest_word = "No Hand"
            gesture_buffer.clear()
            continue

        try:

            gesture, conf = tst.translate_frame(
                results
            )

            gesture = gesture.strip().upper()

            if conf >= CONFIDENCE_THRESHOLD:

                gesture_buffer.append(
                    gesture
                )

                if (
                    len(gesture_buffer)
                    == STABILITY_FRAMES
                    and
                    len(set(gesture_buffer))
                    == 1
                ):

                    latest_word = (
                        gesture_buffer[0]
                    )

            else:
                gesture_buffer.clear()

        except Exception as e:

            latest_word = f"ERROR"

            print(e)

# ──────────────────────────────────────────────
# Flask
# ──────────────────────────────────────────────

@app.route("/")
def home():
    return render_template(
        "index.html"
    )

@app.route("/gesture")
def gesture():

    return jsonify({
        "word": latest_word
    })

# ──────────────────────────────────────────────
# Start
# ──────────────────────────────────────────────

if __name__ == "__main__":

    threading.Thread(
        target=camera_loop,
        daemon=True
    ).start()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
