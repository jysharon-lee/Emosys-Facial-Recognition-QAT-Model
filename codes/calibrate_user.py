import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import time
import os
import csv
import sys

try:
    from picamera2 import Picamera2
except ImportError:
    print("Warning: picamera2 not found. This script is intended to run on the Raspberry Pi.")

# Configuration
GESTURE_LABELS = ["Neutral", "Eye Scratch", "Head Scratch", "Chin Rest",
                  "Nose Scratch", "Neck Rub", "Fidget"]
CALIBRATION_TIME = 10.0  # seconds to record per gesture
PREP_TIME = 3.0          # seconds to prepare before recording

output_dir = os.path.dirname(os.path.abspath(__file__))
csv_file   = os.path.join(output_dir, "calibration_data.csv")

# Feature extraction 
FACE_TARGETS = [0, 2, 5, 7, 8, 9, 10]
HAND_LANDMARKS = [15, 16, 19, 20]
N_FEATURES = 41

def angle_between_cosine(a, b, c):
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-4 or norm_bc < 1e-4:
        return 0.0
    return np.dot(ba, bc) / (norm_ba * norm_bc)

def engineer_features(pts):
    feats = []
    neck = (pts[11] + pts[12]) / 2.0
    for h in HAND_LANDMARKS:
        hand = pts[h]
        for t in FACE_TARGETS:
            feats.append(np.linalg.norm(hand - pts[t]))
        feats.append(np.linalg.norm(hand - neck))
        feats.append(hand[1])
    feats.append(np.linalg.norm(pts[15] - pts[16]))
    feats.append(pts[13][1])
    feats.append(pts[14][1])
    feats.append(angle_between_cosine(pts[11], pts[13], pts[15]))
    feats.append(angle_between_cosine(pts[12], pts[14], pts[16]))
    return np.array(feats, dtype=np.float32)

def extract_features(pose_landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks], dtype=np.float32)
    pts -= pts[0]
    shoulder_dist = np.linalg.norm(pts[11] - pts[12])
    if shoulder_dist > 1e-4:
        pts /= shoulder_dist
    return engineer_features(pts)

# Initialise pose model
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

def main():
    print("Initializing Pi Camera...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    ))
    picam2.start()
    time.sleep(2)

    # Open CSV for writing
    with open(csv_file, mode='w', newline='') as csv_handle:
        writer = csv.writer(csv_handle)
        writer.writerow(['label', 'person_id'] + [f'f_{i}' for i in range(N_FEATURES)])
        
        print("\n=== USER-SPECIFIC CALIBRATION ===")
        print(f"You will hold each of the {len(GESTURE_LABELS)} gestures for {CALIBRATION_TIME} seconds.")
        print("A personalized model will be generated after calibration.\n")
        
        input("Press ENTER when you are ready to begin...")

        for label_idx, gesture_name in enumerate(GESTURE_LABELS):
            # Prep Phase
            prep_end = time.time() + PREP_TIME
            while time.time() < prep_end:
                frame = picam2.capture_array()
                if frame is None: continue
                frame = frame[120:600, 320:960]
                
                rem = int(prep_end - time.time()) + 1
                cv2.putText(frame, f"Get ready for: {gesture_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                cv2.putText(frame, f"Starting in: {rem}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow('Calibration', frame)
                cv2.waitKey(10)

            # Record Phase
            print(f"\nRecording {gesture_name}...")
            record_end = time.time() + CALIBRATION_TIME
            frames_saved = 0
            
            while time.time() < record_end:
                frame = picam2.capture_array()
                if frame is None: continue
                frame = frame[120:600, 320:960]
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = pose_model.detect(mp_image)

                if len(result.pose_landmarks) > 0:
                    for lm in result.pose_landmarks[0]:
                        cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                        
                    features = extract_features(result.pose_landmarks[0])
                    # Person ID is 99 (Calibration User)
                    writer.writerow([label_idx, 99] + features.tolist())
                    frames_saved += 1
                
                rem = int(record_end - time.time()) + 1
                cv2.putText(frame, f"Recording: {gesture_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, f"Time left: {rem}s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, f"Frames saved: {frames_saved}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.imshow('Calibration', frame)
                cv2.waitKey(10)
                
            print(f"Finished recording {gesture_name} ({frames_saved} frames saved).")

    picam2.stop()
    cv2.destroyAllWindows()
    pose_model.close()
    
    print("\nCalibration data collected successfully!")
    print(f"Saved to {csv_file}")
    print("Now run `python finetune_model.py` to generate your personalized model.")

if __name__ == "__main__":
    main()
