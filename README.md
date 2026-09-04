# EmoSys - Real-Time Edge AI Emotion & Environment Analytics
### Built for Raspberry Pi 5 with Pi Camera Module V3
#### Project Built Throughout Internship at SMD Semiconductor
#### Contributor: Aina Qistina | Sharon Lee (Product Development)

---

## What It Is
EmoSys is a highly optimized, real-time Edge AI system designed to understand human state and environmental conditions simultaneously. Engineered specifically for deployment on constrained edge devices like the Raspberry Pi, it runs advanced machine learning models purely on the CPU without requiring massive discrete GPUs. 

## What It Does
The system acts as a live, multi-modal monitoring tool. It watches a video feed and simultaneously detects:
1. **Facial Expressions (Emotions)**
2. **Body Posture (Tension & Relaxation)**
3. **Specific Hand Gestures (Psychological self-adaptors)**
4. **Environmental Climate (Temperature, Humidity, Air Quality)**

As it analyzes the scene, it pushes this data continuously to two places: a local InfluxDB time-series database, and an external SaaS dashboard API (`Emoseeq`), allowing for real-time remote monitoring.

---

## Features & Functionalities

* **User-Specific Gesture Calibration (FaceID-Style):** Because human bodies vary wildly in size and shape, the system includes a 1-minute calibration pipeline (`calibrate_user.py`) that learns the exact skeletal geometry of the user. It fine-tunes a personalized `.tflite` model (`finetune_model.py`) to guarantee near 100% gesture recognition accuracy.
* **Scale-Invariant Feature Engineering:** Hand gestures are analyzed using 41 mathematically engineered distance and cosine-angle features (e.g., elbow angles, wrist-to-nose distances) that are normalized by shoulder width. This makes the AI perfectly accurate whether you are standing 2 feet or 10 feet away from the camera.
* **Multi-Face Tracking & Posture Analysis:** Features a custom `CentroidTracker` capable of tracking multiple faces simultaneously. Uses MediaPipe Pose to calculate a real-time Body Tension Score based on shoulder-to-nose distances.
* **Knowledge Distillation (KD) & QAT:** The core emotion model (MobileNetV2, alpha=0.5) was trained via Knowledge Distillation and Quantization-Aware Training (INT8), allowing it to run at high FPS natively on the Pi CPU.
* **External API Integration:** Seamlessly pushes JSON payloads of the live predictions to external dashboards using `push_module.py`.

---

## Hardware & Software Requirements

### Hardware
* **Raspberry Pi 5** (or Pi 4 with adequate cooling)
* **Raspberry Pi Camera Module 3** (or compatible Pi Camera)
* **Laptop/PC** (Required for the 15-second `finetune_model.py` training step, as the Pi does not run full TensorFlow)
* *(Optional)* I2C Climate Sensors (e.g., BME680)

### Software & Python Dependencies
* **OS:** Raspberry Pi OS (64-bit recommended)
* **Libraries (Pi):** `tflite-runtime`, `opencv-python`, `numpy`, `mediapipe`, `picamera2`, `requests`, `influxdb-client`
* **Libraries (Laptop):** `tensorflow`, `scikit-learn`, `pandas`, `numpy`
* **Database:** InfluxDB v2 (Running locally or remotely)

---

## How to Run

### Step 1: User Calibration (On Raspberry Pi)
Before running inference, you must create a personalized gesture profile so the model understands your specific skeletal structure.
```bash
cd codes
python calibrate_user.py
```
*Follow the on-screen prompts to record 10 seconds of data for each gesture. This will generate a `calibration_data.csv` file.*

### Step 2: Model Training (On Laptop)
Because full TensorFlow is too heavy for the Raspberry Pi, you must transfer `calibration_data.csv` to your PC/Laptop.
```bash
# On your Laptop:
cd codes
python finetune_model.py
```
*This will generate `gesture_model_personal.tflite`. Transfer this file back to the `codes` folder on your Raspberry Pi!*

### Step 3: Run Live Inference (On Raspberry Pi)
Start the main engine. It will automatically detect your personal profile, read the camera, and begin pushing data to the dashboard!
```bash
cd codes
python qat_student_tflite_pi.py
```

---

## Directory & File Guide

### Core Execution Files
* **`codes/qat_student_tflite_pi.py`**: The main execution engine for Raspberry Pi. Captures 640x480 video via `Picamera2`, runs the YuNet face detector, the TFLite Emotion model, the MediaPipe Pose/Gesture model, and manages the background threads for data pushing.
* **`codes/calibrate_user.py`**: The data collection script. Guides the user through a 1-minute pose routine to map their unique physical dimensions to a CSV file.
* **`codes/finetune_model.py`**: The training script (run on a laptop). It reads the CSV, freezes the Conv1D feature extraction layers, trains the dense layers on the user's specific body, and exports the final `gesture_model_personal.tflite`.

### Helper Modules
* **`codes/push_module.py`**: A loose-coupled API client. Packages the emotion, confidence, gesture, and inference speed into a JSON payload and POSTs it to the external `Emoseeq` dashboard.
* **`codes/influxdb_handler.py`**: The time-series database handler. Connects to InfluxDB and pushes data points in a non-blocking background thread.
* **`codes/climate_sensor.py`**: Hardware abstraction layer. Interfaces with I2C climate sensors to read Temperature, Humidity, CO2, VOC, and Particulate Matter (PM).

### Models
* **`qat_student_int8.tflite`**: The highly compressed INT8 quantized MobileNetV2 emotion model.
* **`gesture_model.tflite` / `gesture_model_personal.tflite`**: The Conv1D temporal gesture recognition models.
* **`face_detection_yunet_2023mar.onnx`**: Extremely lightweight face detection model.
* **`pose_landmarker_lite.task`**: MediaPipe's lightweight pose estimation model.
