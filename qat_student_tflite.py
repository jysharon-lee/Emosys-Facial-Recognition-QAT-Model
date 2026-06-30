import cv2
import numpy as np
import tensorflow as tf
Interpreter = tf.lite.Interpreter
import json
import time
from collections import deque
# from picamera2 import Picamera2 # no need pycam anymore since switching to webcam
import matplotlib.pyplot as plt

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
# Setup Webcam
# -----------------------------
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

# smoothing
prediction_buffer = deque(maxlen=30)  # 1s of smoothing at 30 FPS
last_face_seen    = time.time()
FACE_TIMEOUT      = 2.0


# stress tracking
#stress_buffer = deque(maxlen=5)
#prev_stress = 0

#stress_hold_time = 2.0   # seconds to hold stress peak, maybe increase so it will be more realistic
#stress_hold_timer = 0
#stress_peak = 0

# Graph variables
emotion_history   = {label: [] for label in emotion_labels.values()}
time_history      = []

inference_times   = []

# ============================================================
# Helper: draw label with black background
# ============================================================
FACE_PAD        = 30                # pixels of padding around face box
TEXT_PAD        = 2                 # pixels of padding around label text
CONF_THRESHOLD    = 0.22

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


while True:
    
    # Capture frame from Webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    h, w, _ = frame.shape
    yunet.setInputSize((w, h))

    frame_count += 1

    # detect faces every 2 frames
    if frame_count % 2 == 0:
        _, faces = yunet.detect(frame)
        cached_faces = faces
    else:
        faces = cached_faces
        
    # Clear buffer if no face for a while 
    if faces is None:
        if time.time() - last_face_seen > FACE_TIMEOUT:
            prediction_buffer.clear()
    else:
        last_face_seen = time.time()

    if faces is not None:
        for face in faces:
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
            
            inference_times.append(infer_time_ms)

            preds = interpreter.get_tensor(output_details[0]['index'])[0]

            calibration_biases = np.array([
                0.8,   # Angry 
                -0.5,   # Disgust 
                0.0,   # Fear 
                1.0,   # Happy 
                0.8,   # Neutral 
                -0.5,   # Sad 
                0.0    # Surprise 
            ])
            preds = preds + calibration_biases

            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()

            preds = softmax(preds)

            # smoothing
            prediction_buffer.append(preds)
            avg_preds = np.mean(prediction_buffer, axis=0)
            
            # Track Emotion confidence over time for the graph
            time_history.append(time.time() - start)
            for i in range(len(avg_preds)):
                emotion_history[emotion_labels[i]].append(avg_preds[i] * 100)

            # -----------------------------
            # Prediction Output
            # -----------------------------
            top_indices = np.argsort(avg_preds)[-3:][::-1]   # top-3, high =>low
            top1_idx    = top_indices[0]            
            top1_conf = avg_preds[top1_idx]
            top1_label = emotion_labels[top1_idx]

            current_time = time.time()
  
                  
            
            # -----------------------------
            # Logging
            # -----------------------------
            if current_time - last_log_time > log_interval:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                lines = [f"\n[{timestamp}]"]
                lines.append(f"  {'Emotion':<10} {'Conf':>6}  {'Bar'}")
                lines.append(f"  {'-'*9}  {'-'*6}  {'-'*20}")

                sorted_emotions = sorted(
                    [(emotion_labels[i], avg_preds[i]) for i in range(len(avg_preds))],
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

                history_buffer.append(f"[{timestamp}] {top1_label.upper()} {top1_conf*100:.1f}%")

                last_log_time = current_time

            # -----------------------------
            # Stress calculation
            # -----------------------------
            """
            stress_score = 0
            max_stress = 0

            for i, prob in enumerate(avg_preds):
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            # Draw emotion labels
            if top1_conf < CONF_THRESHOLD:
                draw_label(frame, "Uncertain", x1, y1 - 10, 0.8, (0, 165, 255))
            else:
                for rank, idx in enumerate(top_indices):
                    lbl        = emotion_labels[idx]
                    conf_pct   = avg_preds[idx] * 100
                    text       = f"{lbl}: {conf_pct:.1f}%"
                    color      = (255, 0, 255) if rank == 0 else (100, 255, 0)
                    font_scale = 0.8           if rank == 0 else 0.6
                    ty         = y1 - 10 - (rank * 25)
                    draw_label(frame, text, x1, ty, font_scale, color)
                    
            # Inference time box
            infer_text = f"Infer: {infer_time_ms:.1f}ms"
            (tw, th), _ = cv2.getTextSize(infer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            box_x = w - tw - 20
            cv2.rectangle(frame, (box_x - 6, 10), (w - 10, 10 + th + 10), (0, 0, 0), -1)
            cv2.putText(frame, infer_text, (box_x, 10 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

            
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

cap.release()

# -----------------------------
# Plot emotion confidence graph
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle(f"QAT + KD TFLite Trial #{run_count}  —  {run_datetime}", fontsize=10, color="gray")

# --- Line chart ---
ax1.set_title("QAT + KD TFLite Emotion Confidence Over Time", fontsize=13, fontweight="bold")
for label, values in emotion_history.items():
    if len(values) > 0:
        ax1.plot(time_history, values,
                 label=label,
                 color=emotion_colors.get(label, "white"),
                 linewidth=1.5,
                 alpha=0.85)

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Confidence (%)")
ax1.set_ylim(0, 100)
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# --- Bar chart ---
ax2.set_title("QAT + KD TFLite Average Confidence Per Emotion", fontsize=13, fontweight="bold")

avg_per_emotion = {
    label: np.mean(values) if len(values) > 0 else 0
    for label, values in emotion_history.items()
}

sorted_emotions = sorted(avg_per_emotion.items(), key=lambda x: x[1], reverse=True)
labels  = [e[0] for e in sorted_emotions]
values  = [e[1] for e in sorted_emotions]
colors  = [emotion_colors.get(l, "gray") for l in labels]

bars = ax2.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)

# Add percentage label on top of each bar
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f"{val:.1f}%",
             ha="center", va="bottom",
             fontsize=9, fontweight="bold", color="black")

ax2.set_xlabel("Emotion")
ax2.set_ylabel("Average Confidence (%)")
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()

graph_path = f"graph/qat_kd_tflite_emotion_graph_{time.strftime('%Y%m%d_%H%M')}.png"
plt.savefig(graph_path, dpi=150)
print(f"Graph saved to: {graph_path}")
plt.show()

cv2.destroyAllWindows()
