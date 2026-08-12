import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 0: Neutral, 1: Eye Scratch, 2: Head Scratch, 3: Chin Rest, 
# 4: Nose Scratching, 5: Neck Rubbing, 6: Fidgeting
LABELS = {
    0: "Neutral",
    1: "Eye Scratch",
    2: "Head Scratch",
    3: "Chin Rest",
    4: "Nose Scratching",
    5: "Neck Rubbing",
    6: "Fidgeting"
}

csv_file = "gesture_dataset.csv"
is_recording = False
current_label = 0

def extract_features(results):
    # Extract pose
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
        # Normalize to nose (landmark 0)
        nose = pose[0].copy()
        pose = pose - nose
        pose = pose.flatten()
    else:
        pose = np.zeros(33 * 3)
        nose = np.zeros(3)
        
    # Extract left hand
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
        lh = lh - nose # relative to nose too!
        lh = lh.flatten()
    else:
        lh = np.zeros(21 * 3)
        
    # Extract right hand
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        rh = rh - nose
        rh = rh.flatten()
    else:
        rh = np.zeros(21 * 3)
        
    return np.concatenate([pose, lh, rh])

# Initialize CSV
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'f_{i}' for i in range(225)]
        writer.writerow(header)

print("Starting webcam. Please wait...")
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Flip frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # UI overlays
        cv2.putText(image, f"Current Label: {LABELS[current_label]} (Key {current_label})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if is_recording:
            cv2.putText(image, "RECORDING (Press Space to Pause)", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save frame data
            if results.pose_landmarks: # Only save if a person is visible
                features = extract_features(results)
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([current_label] + features.tolist())
        else:
            cv2.putText(image, "PAUSED (Press R to Start)", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Instructions
        cv2.putText(image, "Press 0-6 to change gesture. R to record. Space to pause. Q to quit.", 
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('ML Data Collection', image)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            is_recording = True
        elif key == ord(' '):
            is_recording = False
        elif ord('0') <= key <= ord('6'):
            current_label = key - ord('0')

cap.release()
cv2.destroyAllWindows()
