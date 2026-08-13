"""
Gesture Data Collection Script - RASPBERRY PI VERSION
Uses PiCamera2 + same crop as qat_student_tflite_pi.py

Run once per gesture class:
  python collect_gesture_data_pi.py
  -> Prompts for gesture number, opens Pi camera
  -> R = record, Space = pause, Q = quit & save
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from picamera2 import Picamera2

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

# ── Ask for gesture class and person ID in terminal ───────────────────────────
print("\n--- PI GESTURE DATA COLLECTION ---")
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

try:
    person_id = int(input("Enter person ID (e.g. 1, 2, 3...): ").strip())
except Exception:
    print("Invalid input. Defaulting to person 0.")
    person_id = 0

gesture_name = LABELS[current_label]
output_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gesture Dataset Pi")
os.makedirs(output_dir, exist_ok=True)
csv_file     = os.path.join(output_dir, f"dataset_gesture_{current_label}.csv")
print(f"\nRecording: [{current_label}] {gesture_name}  |  Person: {person_id}")
print(f"Output   : {csv_file}")
print("Press R to start, Space to pause, Q to quit.\n")

# ── Feature extraction (33 pose landmarks × 3 = 99 values) ───────────────────
N_FEATURES = 33 * 3

def extract_features(pose_landmarks):
    """
    Returns a flat numpy array of shape (99,).
    1. Center on nose (landmark 0) for position invariance.
    2. Scale by shoulder width (landmarks 11-12) for body-proportion invariance.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks], dtype=np.float32)
    pts -= pts[0]   # center on nose

    # Scale by shoulder distance so tall/short people produce the same magnitudes
    shoulder_dist = np.linalg.norm(pts[11] - pts[12])
    if shoulder_dist > 1e-4:  # avoid division by zero
        pts /= shoulder_dist

    return pts.flatten()

# ── Initialise pose model ─────────────────────────────────────────────────────
_BaseOptions        = mp_python.BaseOptions
_PoseLandmarker     = mp_vision.PoseLandmarker
_PoseLandmarkerOpts = mp_vision.PoseLandmarkerOptions
_RunningMode        = mp_vision.RunningMode

pose_model = _PoseLandmarker.create_from_options(
    _PoseLandmarkerOpts(
        base_options=_BaseOptions(model_asset_path="pose_landmarker_lite.task"),
        running_mode=_RunningMode.IMAGE,
        num_poses=1,
    )
)

# ── Prepare CSV ───────────────────────────────────────────────────────────────
is_new_file = not os.path.exists(csv_file)
csv_handle  = open(csv_file, mode='a', newline='')
writer      = csv.writer(csv_handle)
if is_new_file:
    writer.writerow(['label', 'person_id'] + [f'f_{i}' for i in range(N_FEATURES)])

# ── Pi Camera ─────────────────────────────────────────────────────────────────
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (1280, 720)}
))
picam2.start()
time.sleep(1)  # let camera warm up
print("Pi Camera started.")

is_recording  = False
frames_saved  = 0

while True:
    frame = picam2.capture_array()
    if frame is None:
        continue

    # ── SAME CROP AS qat_student_tflite_pi.py ──────────────────────────────
    frame = frame[120:600, 320:960]

    h, w = frame.shape[:2]

    # ── Run MediaPipe pose ─────────────────────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = pose_model.detect(mp_image)

    pose_detected = len(result.pose_landmarks) > 0

    # ── Draw landmarks ─────────────────────────────────────────────────────
    if pose_detected:
        for lm in result.pose_landmarks[0]:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    # ── Save data if recording ─────────────────────────────────────────────
    if is_recording and pose_detected:
        features = extract_features(result.pose_landmarks[0])
        writer.writerow([current_label, person_id] + features.tolist())
        frames_saved += 1

    # ── UI Overlay ─────────────────────────────────────────────────────────
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

    cv2.imshow('Pi Gesture Data Collection', frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        is_recording = True
    elif key == ord(' '):
        is_recording = False

# ── Cleanup ───────────────────────────────────────────────────────────────────
picam2.stop()
cv2.destroyAllWindows()
csv_handle.close()
pose_model.close()

print(f"\nDone! Saved {frames_saved} frames to '{csv_file}'.")
