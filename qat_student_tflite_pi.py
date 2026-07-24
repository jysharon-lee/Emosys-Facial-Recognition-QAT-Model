import argparse
import csv
import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import json
import time
from collections import deque
from picamera2 import Picamera2  # Pi Camera support
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from climate_sensor import ClimateReader

# Run counter
run_file = "run_counter/QAT_tflite_run_counter.txt"
try:
    with open(run_file, "r") as f:
        run_count = int(f.read().strip()) + 1
except:
    run_count = 1

with open(run_file, "w") as f:
    f.write(str(run_count))

print(f"Starting Run #{run_count}")

# -----------------------------
# Face Detector
# -----------------------------
yunet = cv2.FaceDetectorYN.create(
    model="face_detection_yunet_2023mar.onnx",
    #model="yunetn_320_qdq_int8.onnx",    # can be use but the detector disappear if got close to the camera. Cant use high (0.5 - 0.7)
    config="",                            # threshold because the model will break in the webcam {not sure for the raspberry pi}
    input_size=(320, 320),                # plus theres the decrease of accuracy after the conversion. 
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)

# -----------------------------
# Labels
# -----------------------------
print ("Loading labels...")
with open("labels_FER.json", "r") as f:
    data = json.load(f)

emotion_labels = {int(k): v for k, v in data.items()}

emotion_labels = {int(k): v for k, v in emotion_labels.items()}

print("Labels loaded!")

# -----------------------------
# TFLITE Model
# -----------------------------
print("Loading QAT TFLite model...")
interpreter = Interpreter(model_path="qat_student_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Grab expected input shape from model (e.g. [1, 96, 96, 3])
input_shape = input_details[0]['shape']
input_h, input_w = input_shape[1], input_shape[2]
print(f"TFLite model loaded! Input size: {input_w}x{input_h}")

# -----------------------------
# Stress mapping
# -----------------------------
"""
stress_map = {
    "angry": 1,
    "fear": 1,
    "sad": 0.8,
    "disgust": 0.7,
    "surprise": 0.5,
    "neutral": 0.2,
    "happy": 0.05
}   # change it up, play with the weight
    # heavy weight in the angry and fear as thats the most show up when someone is stress
"""

# -----------------------------
# Logging
# -----------------------------
log_file = open("log/qat_student_emo_log.txt", "a", encoding="utf-8")
history_buffer = deque(maxlen=3)
last_log_time = time.time()
log_interval = 2.0

# -----------------------------
# Setup Input Source
# -----------------------------
parser = argparse.ArgumentParser(description="EmoSys Inference")
parser.add_argument("--image", type=str, default=None, help="Path to static image file for testing")
args = parser.parse_args()

print("Setting up input source...")
picam2 = None
if args.image:
    print(f"Loading static image: {args.image}")
    static_frame = cv2.imread(args.image)
    if static_frame is None:
        print(f"Error: Could not load image at {args.image}")
        exit()
    cap = None
else:
    print("Opening Pi Camera...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "BGR888"}
    ))
    picam2.start()
    time.sleep(1)  # Let camera warm up
    print("Pi Camera started")
    cap = None  # Not used, but kept for compatibility with cleanup code

# ----------------------------
# Config
# ----------------------------

frame_count = 0
cached_faces = None
start = time.time()
fps_buffer = deque(maxlen=30)

# -----------------------------
# Image Capture Setup
# -----------------------------
capture_folder = os.path.join("Frames", f"Run_{run_count}")
os.makedirs(capture_folder, exist_ok=True)
capture_counter  = 0          # sequential number, resets each run
last_capture_time = time.time()  # track 30-second interval
CAPTURE_INTERVAL  = 30        # seconds between captures

# smoothing (per-face)
face_buffers        = {}   # face_id -> deque of predictions
last_face_seen      = time.time()
FACE_TIMEOUT        = 2.0


# stress tracking
#stress_buffer = deque(maxlen=5)
#prev_stress = 0

#stress_hold_time = 2.0   # seconds to hold stress peak, maybe increase so it will be more realistic
#stress_hold_timer = 0
#stress_peak = 0

# Graph variables (per-face)
face_emotion_history = {}   # face_id -> {label: [values]}
face_time_history    = {}   # face_id -> [timestamps]

inference_times   = []

# ------------------------------------------
# Helper: draw label with black background
# ------------------------------------------
FACE_PAD        = 30                # pixels of padding around face box
TEXT_PAD        = 2                 # pixels of padding around label text
CONF_THRESHOLD    = 0.22
MAX_FACES         = 3               # max faces to process per frame

def draw_label(frame, text, tx, ty, font_scale, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2 if font_scale >= 0.7 else 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(
        frame,
        (tx - TEXT_PAD, ty - th - TEXT_PAD),
        (tx + tw + TEXT_PAD, ty + baseline + TEXT_PAD),
        (0, 0, 0), -1
    )
    cv2.putText(frame, text, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ------------------------------------------------------------------
# Centroid Tracker: assigns persistent IDs to faces across frames
# -----------------------------------------------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=100):
        self.next_id = 0
        self.objects = {}        # id -> (cx, cy)
        self.disappeared = {}    # id -> frames disappeared
        self.max_disappeared = max_disappeared

    def _register(self, centroid):
        fid = self.next_id
        self.objects[fid] = centroid
        self.disappeared[fid] = 0
        self.next_id += 1
        return fid

    def _deregister(self, fid):
        del self.objects[fid]
        del self.disappeared[fid]

    def update(self, bboxes):
        """
        bboxes: list of (x1, y1, x2, y2)
        Returns: list of face_ids aligned 1-to-1 with input bboxes
        """
        input_centroids = [((x1+x2)/2, (y1+y2)/2) for (x1, y1, x2, y2) in bboxes]

        # No detections = mark all existing as disappeared
        if len(input_centroids) == 0:
            for fid in list(self.disappeared.keys()):
                self.disappeared[fid] += 1
                if self.disappeared[fid] > self.max_disappeared:
                    self._deregister(fid)
            return []

        # No existing objects = register all
        if len(self.objects) == 0:
            return [self._register(c) for c in input_centroids]

        # Match existing objects to new detections
        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())

        D = np.zeros((len(obj_centroids), len(input_centroids)))
        for i, oc in enumerate(obj_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.sqrt((oc[0]-ic[0])**2 + (oc[1]-ic[1])**2)

        assignments = {}      
        used_objs = set()
        used_inputs = set()

        # Greedy nearest-first matching
        flat_order = np.argsort(D, axis=None)
        for flat_idx in flat_order:
            row = flat_idx // len(input_centroids)
            col = flat_idx % len(input_centroids)
            if row in used_objs or col in used_inputs:
                continue
            if D[row, col] > 400:      # max pixel distance threshold
                break
            assignments[col] = obj_ids[row]
            self.objects[obj_ids[row]] = input_centroids[col]
            self.disappeared[obj_ids[row]] = 0
            used_objs.add(row)
            used_inputs.add(col)

        # Unmatched existing objects
        for i in range(len(obj_ids)):
            if i not in used_objs:
                fid = obj_ids[i]
                self.disappeared[fid] += 1
                if self.disappeared[fid] > self.max_disappeared:
                    self._deregister(fid)

        # Unmatched new detections
        for j in range(len(input_centroids)):
            if j not in used_inputs:
                assignments[j] = self._register(input_centroids[j])

        return [assignments[j] for j in range(len(input_centroids))]

tracker = CentroidTracker(max_disappeared=50)

# Color palette for face ID labels
FACE_COLORS = [
    (255, 0, 255),   # magenta
    (0, 255, 255),   # cyan
    (0, 255, 0),     # green
]

_BaseOptions        = mp_python.BaseOptions
_PoseLandmarker     = mp_vision.PoseLandmarker
_PoseLandmarkerOpts = mp_vision.PoseLandmarkerOptions
_RunningMode        = mp_vision.RunningMode

pose_model = _PoseLandmarker.create_from_options(
    _PoseLandmarkerOpts(
        base_options=_BaseOptions(model_asset_path="pose_landmarker_lite.task"),
        running_mode=_RunningMode.IMAGE,  
        num_poses=1
    )
)

# Posture state 
posture_label     = "Unknown"
posture_score     = 0.0        # 0.0 = Relaxed, 1.0 = Very Tense
posture_gap_debug = "..."      

# Per-face posture history for graphing
face_posture_history = {}      # face_id -> [posture_score over time]

# ------------------------------------------------------------------
# Environmental Sensing (ClimateReader)
# ------------------------------------------------------------------
climate_sensor = ClimateReader()
climate_sensor.start()

climate_history = {
    'time': [],
    'temp': [],
    'hum': [],
    'co2': [],
    'voc': [],
    'pm': [],
    'discomfort': []
}


os.makedirs("live", exist_ok=True)
csv_file_path = "live/live_data.csv"
# Clear file and write header for the new session
with open(csv_file_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "person_id", "dominant_emotion", 
        "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
        "posture_label", "posture_score",
        "temp", "humidity", "co2", "voc", "pm", "discomfort"
    ])

last_csv_write_time = 0.0

while True:
    
    # Capture frame from Pi Camera or Image
    if args.image:
        frame = static_frame.copy()
        ret = True
        time.sleep(0.033)  
    else:
        frame = picam2.capture_array()
        ret = frame is not None
        if ret:
            # Picamera2 returns RGB, but OpenCV and the model expect BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Crop the center 640x480 to act as a "Digital Zoom". 
            # This makes the face bigger and removes the fish-eye distortion at the edges!
            # frame is currently 1280x720. 
            # y_start = (720 - 480)//2 = 120, x_start = (1280 - 640)//2 = 320
            frame = frame[120:600, 320:960]
        
    if not ret or frame is None:
        print("Failed to grab frame.")
        break

    h, w, _ = frame.shape
    yunet.setInputSize((w, h))

    frame_count += 1

    # MediaPipe Pose 
    if frame_count % 3 == 0:
        frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        pose_results = pose_model.detect(mp_image)

        if pose_results.pose_landmarks:
            lm             = pose_results.pose_landmarks[0]  # first person
            nose           = lm[0]    # Landmark 0  = Nose
            left_shoulder  = lm[11]   # Landmark 11 = Left Shoulder
            right_shoulder = lm[12]   # Landmark 12 = Right Shoulder

            shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2.0
            nose_y         = nose.y

            gap = shoulder_mid_y - nose_y   # positive when relaxed, small when tense
            posture_gap_debug = f"{gap:.3f}"  

            if gap < 0.20:
                posture_label = "Tense"
                posture_score = 1.0
            elif gap < 0.35:
                posture_label = "Slightly Tense"
                posture_score = 0.5
            else:
                posture_label = "Relaxed"
                posture_score = 0.0
        else:
            posture_label     = "Not Detected"
            posture_score     = 0.0
            posture_gap_debug = "N/A"

    # detect faces every 2 frames
    if frame_count % 2 == 0:
        _, faces = yunet.detect(frame)
        cached_faces = faces
    else:
        faces = cached_faces
        
    # Clear per-face buffers if no face for a while
    if faces is None:
        if time.time() - last_face_seen > FACE_TIMEOUT:
            face_buffers.clear()
    else:
        last_face_seen = time.time()

    frame_preds = {}       # face_id -> preds (this frame)
    total_infer_ms = 0

    if faces is not None:
        # Limit to MAX_FACES largest faces
        face_list = faces
        if len(face_list) > MAX_FACES:
            areas = [face_list[i][2] * face_list[i][3] for i in range(len(face_list))]
            top_face_idx = np.argsort(areas)[-MAX_FACES:]
            face_list = face_list[top_face_idx]

        # Compute bounding boxes for tracker
        bboxes = []
        face_rects = []   # (x1, y1, x2, y2) per face for drawing
        for face in face_list:
            x, y, w_box, h_box = face[:4].astype(int)
            x = max(0, x)
            y = max(0, y)
            w_box = min(w_box, w - x)
            h_box = min(h_box, h - y)

            pad = 30
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + w_box + pad)
            y2 = min(h, y + h_box + pad)
            bboxes.append((x1, y1, x2, y2))
            face_rects.append((x1, y1, x2, y2))

        # Get persistent face IDs from tracker
        face_ids = tracker.update(bboxes)

        for idx, (fid, (x1, y1, x2, y2)) in enumerate(zip(face_ids, face_rects)):
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # preprocess
            face_img = cv2.resize(face_crop, (input_w, input_h))
            # frame is already RGB, no second conversion needed
            face_img = face_img.astype("float32") / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            # for tflite model
            scale, zero_point = input_details[0]['quantization']
            face_img = face_img.astype(input_details[0]['dtype'])

            interpreter.set_tensor(input_details[0]['index'], face_img)
            
            infer_start = time.time()
            interpreter.invoke()
            infer_time_ms = (time.time() - infer_start) * 1000
            
            total_infer_ms += infer_time_ms
            inference_times.append(infer_time_ms)

            preds = interpreter.get_tensor(output_details[0]['index'])[0]

            calibration_biases = np.array([
                0.8,   # Angry 
                -0.5,   # Disgust 
                0.0,   # Fear 
                1.0,   # Happy 
                1.0,   # Neutral 
                0.5,   # Sad  (Boosted from 0.0 so it is detected more easily)
                0.0    # Surprise 
            ])
            preds = preds + calibration_biases
            preds = softmax(preds)
            frame_preds[fid] = preds

            # --- Per-face smoothing ---
            if fid not in face_buffers:
                face_buffers[fid] = deque(maxlen=30)
            face_buffers[fid].append(preds)
            avg_preds = np.mean(face_buffers[fid], axis=0)

            # --- Per-face emotion history for graph ---
            if fid not in face_emotion_history:
                face_emotion_history[fid] = {label: [] for label in emotion_labels.values()}
                face_time_history[fid] = []

            face_time_history[fid].append(time.time() - start)
            for i in range(len(avg_preds)):
                face_emotion_history[fid][emotion_labels[i]].append(avg_preds[i] * 100)

            # -----------------------------
            # Per-face Prediction & Drawing
            # -----------------------------
            top_indices = np.argsort(avg_preds)[-3:][::-1]   # top-3, high =>low
            top1_idx    = top_indices[0]
            top1_conf   = avg_preds[top1_idx]
            top1_label  = emotion_labels[top1_idx]

            # -----------------------------
            # Stress calculation
            # -----------------------------
            """
            stress_score = 0
            max_stress = 0

            for i, prob in enumerate(preds):
                label = emotion_labels[i]
                weight = stress_map[label]
                
                stress_score += prob * weight
                max_stress += weight

                stress_percent = stress_score * 100

                stress_buffer.append(stress_percent)
                stress_percent = np.mean(stress_buffer)
                
                current_time = time.time()

                # If stress increases → reset hold
                if stress_percent > stress_peak:
                    stress_peak = stress_percent
                    stress_hold_timer = current_time

                decay_speed = 0.5  # adjust (lower = slower decay)

                if current_time - stress_hold_timer > stress_hold_time:
                    stress_peak *= decay_speed
                    stress_percent = stress_peak

                stress_percent = np.power(stress_percent / 100, 1.2) * 100

                stress_percent = np.clip(stress_percent, 0, 100)

            # stress level
            if stress_percent < 30:
                stress_level = "Relaxed"
                stress_color = (0, 200, 0)
            elif stress_percent < 60:
                stress_level = "Moderate"
                stress_color = (0, 165, 255)
            else:
                stress_level = "High"
                stress_color = (0, 0, 255)

            # Stress trend
            threshold = 2
            if stress_percent > prev_stress + threshold:
                trend = "up"
            elif stress_percent < prev_stress - threshold:
                trend = "down"
            else:
                trend = "-"

            prev_stress = stress_percent
            """
          
            # DRAW FACE BOX
            box_color = FACE_COLORS[idx % len(FACE_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Face ID label
            draw_label(frame, f"Face #{fid}", x2 - 70, y2 + 18, 0.5, box_color)

            # Posture label + raw gap debug value below face box
            draw_label(frame, f"Posture: {posture_label}  [gap={posture_gap_debug}]", x1, y2 + 38, 0.5, (255, 255, 0))

            # Draw emotion labels per face
            if top1_conf < CONF_THRESHOLD:
                draw_label(frame, "Uncertain", x1, y1 - 10, 0.8, (0, 165, 255))
            else:
                for rank, eidx in enumerate(top_indices):
                    lbl        = emotion_labels[eidx]
                    conf_pct   = avg_preds[eidx] * 100
                    text       = f"{lbl}: {conf_pct:.1f}%"
                    color      = box_color if rank == 0 else (100, 255, 0)
                    font_scale = 0.8       if rank == 0 else 0.6
                    ty         = y1 - 10 - (rank * 25)
                    draw_label(frame, text, x1, ty, font_scale, color)

            # Store posture history for graphing
            if fid not in face_posture_history:
                face_posture_history[fid] = []
            face_posture_history[fid].append(posture_score)

            """
            # -----------------------------
            # Stress bar
            # -----------------------------
            
            bar_x, bar_y = x1, y2 + 10
            bar_w, bar_h = 150, 12
            filled = int((stress_percent / 100) * bar_w)

            cv2.rectangle(frame,
                          (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h),
                          (50, 50, 50), -1)

            cv2.rectangle(frame,
                          (bar_x, bar_y),
                          (bar_x + filled, bar_y + bar_h),
                          stress_color, -1)

            cv2.putText(frame,
                        f"{stress_level} {stress_percent:.1f}% {trend}",
                        (bar_x, bar_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        stress_color, 1)
            """
    else:
        tracker.update([])   # keep tracker ticking when no faces

    # ---------------------------------------------------------
    # Per-face logging (outside face loop)
    # ---------------------------------------------------------
    if frame_preds:
        current_time = time.time()

        if current_time - last_log_time > log_interval:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            for fid, preds in frame_preds.items():
                avg_p = np.mean(face_buffers[fid], axis=0) if fid in face_buffers else preds
                top1_idx   = np.argmax(avg_p)
                top1_label = emotion_labels[top1_idx]
                top1_conf  = avg_p[top1_idx]

                lines = [f"\n[{timestamp}] Face #{fid}"]
                lines.append(f"  {'Emotion':<10} {'Conf':>6}  {'Bar'}")
                lines.append(f"  {'-'*9}  {'-'*6}  {'-'*20}")

                sorted_emotions = sorted(
                    [(emotion_labels[i], avg_p[i]) for i in range(len(avg_p))],
                    key=lambda x: x[1],
                    reverse=True
                )

                for label, conf in sorted_emotions:
                    bar_len = int(conf * 20)
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    marker = " ◄" if label == top1_label else ""
                    lines.append(f"  {label:<10} {conf*100:>5.1f}%  {bar}{marker}")

                log_text = "\n".join(lines)
                log_file.write(log_text + "\n")
                log_file.flush()

                history_buffer.append(f"[{timestamp}] F#{fid}: {top1_label.upper()} {top1_conf*100:.1f}%")

            last_log_time = current_time

        # Inference time box 
        n_faces_now = len(frame_preds)
        infer_text = f"Infer: {total_infer_ms:.1f}ms ({n_faces_now}f)"
        (tw, th_text), _ = cv2.getTextSize(infer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        box_x = w - tw - 20
        cv2.rectangle(frame, (box_x - 6, 10), (w - 10, 10 + th_text + 10), (0, 0, 0), -1)
        cv2.putText(frame, infer_text, (box_x, 10 + th_text + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        

    # Read Environment Data & Draw UI
    t_val, h_val, co2_val, voc_val, pm_val = climate_sensor.get_readings()
    env_discomfort = 0.0

    if t_val is not None:
        # Simple Environmental Discomfort Score (0-100%)
        # Penalties for:
        # CO2 > 800ppm (+0 to 40)
        co2_penalty = min(40.0, max(0.0, (co2_val - 800) / 10.0))
        # VOC > 100 (+0 to 30)
        voc_penalty = min(30.0, max(0.0, (voc_val - 100) / 2.0))
        # PM > 25 (+0 to 30)
        pm_penalty = min(30.0, max(0.0, (pm_val - 25) * 2.0))
        
        env_discomfort = min(100.0, co2_penalty + voc_penalty + pm_penalty)

        # Log history for graph
        if frame_count % 10 == 0:  # Save every 10 frames to avoid huge arrays
            climate_history['time'].append(time.time() - start)
            climate_history['temp'].append(t_val)
            climate_history['hum'].append(h_val)
            climate_history['co2'].append(co2_val)
            climate_history['voc'].append(voc_val)
            climate_history['pm'].append(pm_val)
            climate_history['discomfort'].append(env_discomfort)

        # Draw Environment UI in Top Right
        env_x = w - 180
        env_y = 65
        cv2.rectangle(frame, (env_x - 10, env_y - 20), (w - 10, env_y + 115), (0, 0, 0), -1)
        cv2.putText(frame, "Environment:", (env_x, env_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Temp: {t_val:.1f}C", (env_x, env_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"Hum:  {h_val:.1f}%", (env_x, env_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"CO2:  {co2_val:.0f} ppm", (env_x, env_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"VOC:  {voc_val:.0f}", (env_x, env_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"PM:   {pm_val:.0f}", (env_x, env_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Color discomfort red if high
        color_disc = (0, 0, 255) if env_discomfort > 50 else (0, 255, 0)
        cv2.putText(frame, f"Disc: {env_discomfort:.1f}%", (env_x, env_y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_disc, 1)

    # History UI
    history_x = 10
    history_y = 60

    cv2.putText(frame, "History:", (history_x, history_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for i, text in enumerate(reversed(history_buffer)):
        cv2.putText(frame,
                    text,
                    (history_x, history_y + 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 200, 200), 1)
            
    # FPS
    fps = frame_count / (time.time() - start)
    fps_buffer.append(fps)
    cv2.putText(frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255), 2)
                
    # ---------------------------------------------------------
    # Streamlit CSV Writing (1x per second)
    # ---------------------------------------------------------
    current_time_end = time.time()
    if current_time_end - last_csv_write_time >= 1.0:
        last_csv_write_time = current_time_end
        
        # Default environment values if none exist yet
        write_t = t_val if t_val is not None else 0.0
        write_h = h_val if t_val is not None else 0.0
        write_c = co2_val if t_val is not None else 0.0
        write_v = voc_val if t_val is not None else 0.0
        write_p = pm_val if t_val is not None else 0.0
        write_d = env_discomfort if t_val is not None else 0.0

        with open(csv_file_path, "a", newline="") as f:
            writer = csv.writer(f)
            
            if not frame_preds:
                # No faces detected, but still write environment data
                writer.writerow([
                    current_time_end, "None", "None", 
                    0, 0, 0, 0, 0, 0, 0,
                    posture_label, posture_score,
                    write_t, write_h, write_c, write_v, write_p, write_d
                ])
            else:
                for fid, preds in frame_preds.items():
                    # Calculate smoothed predictions for the CSV
                    avg_p = np.mean(face_buffers[fid], axis=0) if fid in face_buffers else preds
                    top1_idx = np.argmax(avg_p)
                    top_emotion = emotion_labels[top1_idx]
                    
                    writer.writerow([
                        current_time_end, f"Person {fid}", top_emotion, 
                        avg_p[0], avg_p[1], avg_p[2], avg_p[3], avg_p[4], avg_p[5], avg_p[6],
                        posture_label, posture_score,
                        write_t, write_h, write_c, write_v, write_p, write_d
                    ])

    cv2.imshow("Yunet + MB-V2 model(QAT)", frame)

    # ---------------------------------------------------------
    # Periodic Frame Capture (every 30 seconds, face required)
    # ---------------------------------------------------------
    current_time_capture = time.time()
    if (current_time_capture - last_capture_time >= CAPTURE_INTERVAL) and frame_preds:
        capture_counter += 1
        filename    = f"frame_{capture_counter:03d}.jpg"
        filepath    = os.path.join(capture_folder, filename)
        cv2.imwrite(filepath, frame)
        print(f"Frame saved: {filepath}")
        last_capture_time = current_time_capture

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        print("Ending the program...")
        print("Output dtype:", output_details[0]['dtype'])
        print("Output quantization (scale, zp):", output_details[0]['quantization'])

        raw_output = interpreter.get_tensor(output_details[0]['index'])[0]
        print("Raw output:", raw_output)
        print("Sum:", raw_output.sum())
        break

climate_sensor.stop()

# Cleanup Pi Camera
if picam2 is not None:
    picam2.stop()

end = time.time()
avg_fps = np.mean(fps_buffer)
total_time = time.time() - start
print(f"\n---- Trial Summary ----")
print(f"\nTotal runtime       : {total_time:.2f}s")
print(f"Average FPS           : {avg_fps:.1f}")
print(f"Total frames processed: {frame_count}")

if inference_times:
    avg_infer   = np.mean(inference_times)
    min_infer   = np.min(inference_times)
    max_infer   = np.max(inference_times)
    std_infer   = np.std(inference_times)
    p95_infer   = np.percentile(inference_times, 95)
    
    print(f"\n---- Inference Speed Summary ----")
    print(f"Total inferences: {len(inference_times)}")
    print(f"Average         : {avg_infer:.2f} ms")
    print(f"Min             : {min_infer:.2f} ms")
    print(f"Max             : {max_infer:.2f} ms")
    print(f"Std dev         : {std_infer:.2f} ms")
    print(f"95th Percentile : {p95_infer:.2f} ms")
    
    with open("log/qat_student_tflite_inference_speed_log.txt", "a", encoding="utf-8") as speed_log:
        speed_log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Trial #{run_count}\n")
        speed_log.write(f"  Total runtime          : {total_time:.2f}s\n")
        speed_log.write(f"  Average FPS            : {avg_fps:.1f}\n")
        speed_log.write(f"  Total frames processed : {frame_count}\n\n")
        speed_log.write(f"  Total inference        : {len(inference_times)}\n")
        speed_log.write(f"  Avg                    : {avg_infer:.2f} ms\n")
        speed_log.write(f"  Min                    : {min_infer:.2f} ms\n")
        speed_log.write(f"  Max                    : {max_infer:.2f} ms\n")
        speed_log.write(f"  Std Dev                : {std_infer:.2f} ms\n")
        speed_log.write(f"  P95                    : {p95_infer:.2f} ms\n")
log_file.close()

cv2.destroyAllWindows()
pose_model.close()
print("Session complete. All data saved to live/live_data.csv")
