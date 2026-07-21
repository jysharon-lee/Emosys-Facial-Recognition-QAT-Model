# EmoSys - Real-Time Edge AI Emotion & Environment Analytics
### Built for Raspberry Pi 5 with Pi Camera Module 3

EmoSys is a highly optimized Edge AI system that analyzes **human emotions**, **body posture**, and **environmental climate** in real-time. It was specifically engineered for deployment on constrained edge devices like the Raspberry Pi, utilizing a custom-trained compressed MobileNetV2 architecture and a dedicated live Streamlit dashboard.

---

## 🌟 Key Features

* **Multi-Face Tracking & Posture Analysis:** Features a custom `CentroidTracker` capable of tracking multiple faces simultaneously. It also uses MediaPipe Pose to calculate a real-time Body Tension Score based on shoulder-to-nose distances.
* **Environmental Climate Sensing:** Integrates directly with hardware climate sensors via I2C to read Temperature, Humidity, CO2, VOC, and Particulate Matter (PM), calculating a live Environmental Discomfort Index.
* **Knowledge Distillation (KD) & QAT:** The core emotion model (MobileNetV2, alpha=0.5) was trained via Knowledge Distillation and Quantization-Aware Training (INT8), allowing it to run at high FPS purely on the Pi's CPU without catastrophic accuracy loss.
* **Digital Zoom & Distortion Correction:** The Pi camera capture logic utilizes an active sensor crop (1280x720 down to 640x480) to optically zoom the frame and surgically eliminate wide-angle fish-eye distortion.
* **Live SaaS-Style Dashboard:** The AI inference script runs completely decoupled from visualization. It writes data to a local CSV every second, which is instantly read by a lightweight Streamlit web app, allowing you to view real-time charts from any device on the network.

---

## ⚙️ Requirements

To run this system, you will need a Raspberry Pi (ideally Pi 5) with a Pi Camera Module 3, and the following packages installed in your virtual environment:

```bash
pip install tflite-runtime opencv-python numpy mediapipe picamera2
pip install streamlit streamlit-autorefresh plotly pandas
```

---

## 🚀 How to Run

The system is split into two parts: The **AI Engine** and the **Dashboard Visualizer**. You will need to open two separate terminal windows on your Raspberry Pi.

### 1. Start the AI Engine (Terminal 1)
Activate your virtual environment and run the main inference script.
```bash
cd ~/Downloads/'EmoSys - KD N QAT'
source env/bin/activate
python qat_student_tflite_pi.py
```
*A window will open showing the camera feed with bounding boxes. To safely exit, press `Ctrl+C`.*

### 2. Start the Dashboard (Terminal 2)
In a new terminal window, activate the same environment and launch Streamlit:
```bash
cd ~/Downloads/'EmoSys - KD N QAT'
source env/bin/activate
streamlit run dashboard.py
```

### 3. View the Analytics
Once Streamlit starts, it will display a local Network URL (e.g., `http://192.168.1.X:8501`). 
Open a web browser on your **Windows PC** or phone (while connected to the same Wi-Fi), enter that URL, and you will see the live, real-time analytics dashboard!

---

## 📂 Repository Structure

### Core & Inference
* **`qat_student_tflite_pi.py`**: The main execution engine. Captures video via `Picamera2`, runs YuNet for face detection, TFLite for emotion classification, MediaPipe for posture, and `ClimateReader` for sensors. Writes output to `live/live_data.csv`.
* **`dashboard.py`**: The Streamlit web application. Provides a premium, SaaS-style clean UI using Plotly charts for live data visualization.
* **`climate_sensor.py`**: The hardware abstraction layer that communicates with the physical I2C climate sensors.

### Models
* **`qat_student_int8.tflite`**: The highly compressed INT8 quantized MobileNetV2 emotion model.
* **`face_detection_yunet_2023mar.onnx`**: Extremely lightweight face detection model.
* **`pose_landmarker_lite.task`**: MediaPipe's lightweight pose estimation model.

### Output & Logging
* **`live/`**: Contains `live_data.csv` which acts as the real-time bridge between the AI engine and the dashboard.
* **`log/`**: Stores timestamped text files containing raw emotion confidence data and inference speed diagnostics.
* **`run_counter/`**: Keeps track of the total number of trial sessions run by the system.
