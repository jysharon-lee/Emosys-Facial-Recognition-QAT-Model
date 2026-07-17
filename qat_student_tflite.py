import argparse
import cv2
import numpy as np
import tensorflow as tf
Interpreter = tf.lite.Interpreter
import json
import time
from collections import deque
# from picamera2 import Picamera2 # no need pycam anymore since switching to webcam
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

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
if args.image:
    print(f"Loading static image: {args.image}")
    static_frame = cv2.imread(args.image)
    if static_frame is None:
        print(f"Error: Could not load image at {args.image}")
        exit()
    cap = None
else:
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    print("Camera started")

# ----------------------------
# Config
# ----------------------------

frame_count = 0
cached_faces = None
start = time.time()
fps_buffer = deque(maxlen=30)

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
    def __init__(self, max_disappeared=50):
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

        # No detections — mark all existing as disappeared
        if len(input_centroids) == 0:
            for fid in list(self.disappeared.keys()):
                self.disappeared[fid] += 1
                if self.disappeared[fid] > self.max_disappeared:
                    self._deregister(fid)
            return []

        # No existing objects — register all
        if len(self.objects) == 0:
            return [self._register(c) for c in input_centroids]

        # Match existing objects to new detections
        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())

        D = np.zeros((len(obj_centroids), len(input_centroids)))
        for i, oc in enumerate(obj_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.sqrt((oc[0]-ic[0])**2 + (oc[1]-ic[1])**2)

        assignments = {}      # input_idx -> face_id
        used_objs = set()
        used_inputs = set()

        # Greedy nearest-first matching
        flat_order = np.argsort(D, axis=None)
        for flat_idx in flat_order:
            row = flat_idx // len(input_centroids)
            col = flat_idx % len(input_centroids)
            if row in used_objs or col in used_inputs:
                continue
            if D[row, col] > 150:      # max pixel distance threshold
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

# ------------------------------------------------------------------
# MediaPipe Pose: Specialist 2 — Body Language Tracking
# (Using Tasks API for MediaPipe 0.10.x / Python 3.13 compatibility)
# ------------------------------------------------------------------
_BaseOptions        = mp_python.BaseOptions
_PoseLandmarker     = mp_vision.PoseLandmarker
_PoseLandmarkerOpts = mp_vision.PoseLandmarkerOptions
_RunningMode        = mp_vision.RunningMode

pose_model = _PoseLandmarker.create_from_options(
    _PoseLandmarkerOpts(
        base_options=_BaseOptions(model_asset_path="pose_landmarker_lite.task"),
        running_mode=_RunningMode.IMAGE,   # IMAGE = synchronous, one frame at a time
        num_poses=1
    )
)

# Posture state (updated per frame, shared across faces in the scene)
posture_label = "Unknown"
posture_score = 0.0            # 0.0 = Relaxed, 1.0 = Very Tense

# Per-face posture history for graphing
face_posture_history = {}      # face_id -> [posture_score over time]


while True:
    
    # Capture frame from Webcam or Image
    if args.image:
        frame = static_frame.copy()
        ret = True
        time.sleep(0.033)  # simulate ~30fps so it doesn't run infinitely fast
    else:
        ret, frame = cap.read()
        
    if not ret:
        print("Failed to grab frame.")
        break

    h, w, _ = frame.shape
    yunet.setInputSize((w, h))

    frame_count += 1

    # ------------------------------------------------------------------
    # Specialist 2: MediaPipe Pose — run every 3rd frame to save CPU
    # ------------------------------------------------------------------
    if frame_count % 3 == 0:
        frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        pose_results = pose_model.detect(mp_image)

        if pose_results.pose_landmarks:
            lm             = pose_results.pose_landmarks[0]  # first person
            nose           = lm[0]    # Landmark 0  = Nose
            left_shoulder  = lm[11]   # Landmark 11 = Left Shoulder
            right_shoulder = lm[12]   # Landmark 12 = Right Shoulder

            # Shoulder midpoint Y (all values normalised 0.0–1.0)
            # NOTE: Y=0 is TOP of screen. Shoulders are BELOW nose, so
            # shoulder_mid_y > nose_y. A smaller gap means shoulders
            # have risen toward the ears = tension.
            shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2.0
            nose_y         = nose.y

            gap = shoulder_mid_y - nose_y   # positive when relaxed, small when tense
            posture_gap_debug = f"{gap:.3f}"  # store for on-screen debug display

            if gap < 0.10:
                posture_label = "Tense"
                posture_score = 1.0
            elif gap < 0.18:
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
                0.0,   # Sad 
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
            # -----------------------------
            # DRAW FACE BOX
            # -----------------------------
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

        # Inference time box (total across all faces)
        n_faces_now = len(frame_preds)
        infer_text = f"Infer: {total_infer_ms:.1f}ms ({n_faces_now}f)"
        (tw, th_text), _ = cv2.getTextSize(infer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        box_x = w - tw - 20
        cv2.rectangle(frame, (box_x - 6, 10), (w - 10, 10 + th_text + 10), (0, 0, 0), -1)
        cv2.putText(frame, infer_text, (box_x, 10 + th_text + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            
    # -----------------------------
    # History UI
    # -----------------------------
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

    cv2.imshow("Yunet + MB-V2 model(QAT)", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        print("Ending the program...")
        print("Output dtype:", output_details[0]['dtype'])
        print("Output quantization (scale, zp):", output_details[0]['quantization'])
        # If dtype=int8 and scale != 0.0 → you need dequantization
        
        raw_output = interpreter.get_tensor(output_details[0]['index'])[0]
        print("Raw output:", raw_output)
        # Logits look like:  [ 3.2  -1.5   0.8  -2.1  ...]  (arbitrary range)
        # Probabilities look like: [0.65  0.12  0.08  0.05 ...]  (0–1, sums to ~1)
        print("Sum:", raw_output.sum())  # ~1.0 = already softmax; anything else = raw logits
        
        
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
        break

if cap is not None:
    cap.release()

# -----------------------------
# Plot emotion confidence graph (per-face)
# -----------------------------
emotion_colors = {
    "angry":    "#FF4444",
    "disgust":  "#AA44FF",
    "fear":     "#FF8800",
    "happy":    "#FFD700",
    "neutral":  "#44AAFF",
    "sad":      "#0014F7",
    "surprise": "#44FF88",
}

run_datetime = time.strftime("%A, %d %B %Y  |  %H:%M:%S")

# Sort face IDs for consistent ordering
tracked_faces = sorted(face_emotion_history.keys())
n_faces = max(len(tracked_faces), 1)

# +1 extra row for the posture graph
fig, axes = plt.subplots(n_faces + 1, 2, figsize=(18, 6 * (n_faces + 1)), squeeze=False)
fig.suptitle(f"QAT + KD TFLite Trial #{run_count}  \u2014  {run_datetime}", fontsize=12, color="gray")

for row, fid in enumerate(tracked_faces):
    ax_line = axes[row][0]
    ax_bar  = axes[row][1]

    emo_hist = face_emotion_history[fid]
    t_hist   = face_time_history[fid]

    # --- Line chart ---
    ax_line.set_title(f"Face #{fid} — Emotion Confidence Over Time", fontsize=13, fontweight="bold")
    for label, values in emo_hist.items():
        if len(values) > 0:
            ax_line.plot(t_hist, values,
                         label=label,
                         color=emotion_colors.get(label, "white"),
                         linewidth=1.5,
                         alpha=0.85)

    ax_line.set_xlabel("Time (s)")
    ax_line.set_ylabel("Confidence (%)")
    ax_line.set_ylim(0, 100)
    ax_line.legend(loc="upper right", fontsize=8)
    ax_line.grid(True, alpha=0.3)

    # --- Bar chart ---
    ax_bar.set_title(f"Face #{fid} — Average Confidence Per Emotion", fontsize=13, fontweight="bold")

    avg_per_emotion = {
        label: np.mean(values) if len(values) > 0 else 0
        for label, values in emo_hist.items()
    }

    sorted_emotions = sorted(avg_per_emotion.items(), key=lambda x: x[1], reverse=True)
    labels_list = [e[0] for e in sorted_emotions]
    values_list = [e[1] for e in sorted_emotions]
    colors_list = [emotion_colors.get(l, "gray") for l in labels_list]

    bars = ax_bar.bar(labels_list, values_list, color=colors_list, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values_list):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="black")

    ax_bar.set_xlabel("Emotion")
    ax_bar.set_ylabel("Average Confidence (%)")
    ax_bar.set_ylim(0, 100)
    ax_bar.grid(True, alpha=0.3, axis="y")

# Handle case where no faces were detected at all
if len(tracked_faces) == 0:
    axes[0][0].set_title("No faces detected", fontsize=13)
    axes[0][1].set_title("No faces detected", fontsize=13)

# ------------------------------------------------------------------
# Posture Graph Row (last row, spans both columns)
# ------------------------------------------------------------------
ax_posture_line = axes[n_faces][0]
ax_posture_bar  = axes[n_faces][1]

posture_color_map = {0.0: "#44FF88", 0.5: "#FFD700", 1.0: "#FF4444"}

for fid in tracked_faces:
    p_hist = face_posture_history.get(fid, [])
    if len(p_hist) > 0:
        t_hist = face_time_history[fid][:len(p_hist)]
        color  = FACE_COLORS[fid % len(FACE_COLORS)]
        # Convert BGR tuple to hex for matplotlib
        hex_color = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])
        ax_posture_line.plot(
            t_hist, p_hist,
            label=f"Face #{fid} Posture",
            color=hex_color,
            linewidth=1.5,
            alpha=0.85
        )

        # Bar: average posture score
        avg_posture = np.mean(p_hist)
        ax_posture_bar.bar(
            f"Face #{fid}",
            avg_posture * 100,
            color=hex_color,
            edgecolor="white",
            linewidth=0.5
        )
        ax_posture_bar.text(
            f"Face #{fid}",
            avg_posture * 100 + 1,
            f"{avg_posture * 100:.1f}%",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold"
        )

ax_posture_line.set_title("Body Language — Posture Tension Over Time", fontsize=13, fontweight="bold")
ax_posture_line.set_xlabel("Time (s)")
ax_posture_line.set_ylabel("Tension Score (0=Relaxed, 1=Tense)")
ax_posture_line.set_ylim(-0.1, 1.2)
ax_posture_line.axhline(y=0.5, color="yellow", linestyle="--", alpha=0.5, label="Slight Tension Threshold")
ax_posture_line.axhline(y=1.0, color="red",    linestyle="--", alpha=0.5, label="High Tension Threshold")
ax_posture_line.legend(loc="upper right", fontsize=8)
ax_posture_line.grid(True, alpha=0.3)

ax_posture_bar.set_title("Body Language — Average Posture Tension Per Person", fontsize=13, fontweight="bold")
ax_posture_bar.set_xlabel("Person")
ax_posture_bar.set_ylabel("Average Tension (%)")
ax_posture_bar.set_ylim(0, 110)
ax_posture_bar.grid(True, alpha=0.3, axis="y")

plt.tight_layout()

graph_path = f"graph/qat_kd_tflite_emotion_graph_{time.strftime('%Y%m%d_%H%M')}.png"
plt.savefig(graph_path, dpi=150)
print(f"Graph saved to: {graph_path}")
plt.show()

cv2.destroyAllWindows()
pose_model.close()
