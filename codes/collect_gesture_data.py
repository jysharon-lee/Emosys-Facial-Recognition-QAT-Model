"""
Gesture Data Collection Script
Uses the exact same MediaPipe Tasks API as qat_student_tflite_pi.py

Run once per gesture class:
  python collect_gesture_data.py
  -> Prompts for gesture number, opens webcam
  -> R = record, Space = pause, Q = quit & save

Output: dataset_gesture_<N>.csv
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Labels ────────────────────────────────────────────────────────────────────
LABELS = {
    0: "Neutral",
    1: "Eye Scratch",
    2: "Head Scratch",
    3: "Chin Rest",
    4: "Nose Scratching",
    5: "Neck Rubbing",
    6: "Fidgeting",
}

# ── Ask for gesture class in terminal ─────────────────────────────────────────
print("\n--- GESTURE DATA COLLECTION ---")
print("Which gesture do you want to record?\n")
for k, v in LABELS.items():
    print(f"  {k}: {v}")
print()
try:
    current_label = int(input("Enter gesture number (0-6): ").strip())
    if current_label not in LABELS:
        raise ValueError
except Exception:
    print("Invalid input. Defaulting to 0 (Neutral).")
    current_label = 0

gesture_name = LABELS[current_label]
csv_file     = f"dataset_gesture_{current_label}.csv"
print(f"\nRecording: [{current_label}] {gesture_name}")
print(f"Output   : {csv_file}")
print("Press R to start, Space to pause, Q to quit.\n")

# ── Feature extraction (33 pose landmarks × 3 = 99 values) ───────────────────
N_FEATURES = 33 * 3   # x, y, z for each pose landmark

def extract_features(pose_landmarks):
    """
    Returns a flat numpy array of shape (99,).
    All coordinates are shifted so the nose (landmark 0) is at 0,0,0.
    This makes the data invariant to camera distance and seating position.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks], dtype=np.float32)
    pts -= pts[0]   # subtract nose position
    return pts.flatten()

# ── Initialise pose model (same .task file as the Pi script) ──────────────────
_BaseOptions        = mp_python.BaseOptions
_PoseLandmarker     = mp_vision.PoseLandmarker
_PoseLandmarkerOpts = mp_vision.PoseLandmarkerOptions
_RunningMode        = mp_vision.RunningMode

TASK_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "pose_landmarker_lite.task"
)

pose_model = _PoseLandmarker.create_from_options(
    _PoseLandmarkerOpts(
        base_options=_BaseOptions(model_asset_path=TASK_FILE),
        running_mode=_RunningMode.IMAGE,
        num_poses=1,
    )
)

# ── Prepare CSV ───────────────────────────────────────────────────────────────
is_new_file = not os.path.exists(csv_file)
csv_handle  = open(csv_file, mode='a', newline='')
writer      = csv.writer(csv_handle)
if is_new_file:
    writer.writerow(['label'] + [f'f_{i}' for i in range(N_FEATURES)])

# ── Webcam loop ───────────────────────────────────────────────────────────────
is_recording  = False
frames_saved  = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    csv_handle.close()
    exit(1)


while True:
    ret, frame = cap.read()
    if not ret:
        print("WARNING: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)                          # mirror view

    # ── Run MediaPipe pose on this frame ──────────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = pose_model.detect(mp_image)

    pose_detected = len(result.pose_landmarks) > 0

    # ── Draw skeleton (manual OpenCV draw, no mp.solutions needed) ─────────
    if pose_detected:
        h, w = frame.shape[:2]
        lm_proto = result.pose_landmarks[0]
        for lm in lm_proto:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    # ── Save data if recording ────────────────────────────────────────────────
    if is_recording and pose_detected:
        features = extract_features(result.pose_landmarks[0])
        writer.writerow([current_label] + features.tolist())
        frames_saved += 1

    # ── UI Overlay ────────────────────────────────────────────────────────────
    label_text  = f"Gesture: [{current_label}] {gesture_name}"
    state_text  = "RECORDING" if is_recording else "PAUSED"
    state_color = (0, 0, 255) if is_recording else (255, 80, 0)
    no_pose_txt = "" if pose_detected else "  <NO POSE DETECTED>"

    cv2.putText(frame, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    cv2.putText(frame, f"{state_text}  |  Frames saved: {frames_saved}{no_pose_txt}",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)
    cv2.putText(frame, "R = Record   Space = Pause   Q = Quit & Save",
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    cv2.imshow('Gesture Data Collection', frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        is_recording = True
    elif key == ord(' '):
        is_recording = False

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
csv_handle.close()
pose_model.close()

print(f"\nDone! Saved {frames_saved} frames to '{csv_file}'.")
