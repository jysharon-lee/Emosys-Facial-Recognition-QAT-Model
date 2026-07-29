# EmoSys - Real-Time Edge AI Emotion & Environment Analytics
### Built for Raspberry Pi 5 with Pi Camera Module 3 & PC Webcams
#### Project Built Throughout Internship at SMD Semiconductor
#### Contributor: Aina Qistina | Sharon Lee (Product Development)

EmoSys is a highly optimized Edge AI system that analyzes **human emotions**, **body posture**, **hand gestures**, and **environmental climate** in real-time. It is engineered for deployment on constrained edge devices like the Raspberry Pi, utilizing a custom-trained compressed MobileNetV2 architecture, InfluxDB for time-series logging, and a dedicated live Streamlit dashboard. It also supports standard Webcams for testing on PC.

---

## 🌟 Key Features

* **Multi-Face Tracking & Posture Analysis:** Features a custom `CentroidTracker` capable of tracking multiple faces simultaneously. Uses MediaPipe Pose to calculate a real-time Body Tension Score based on shoulder-to-nose distances.
* **Hand Gesture Recognition:** Utilizes MediaPipe Pose landmarks combined with a heuristic spatial-temporal classifier to detect psychological self-adaptor gestures like *Chin Rest*, *Face Touch*, *Forehead Rub*, *Mouth Cover*, and *Head Scratch*.
* **Environmental Climate Sensing (Pi Only):** Integrates directly with hardware climate sensors via I2C to read Temperature, Humidity, CO2, VOC, and Particulate Matter (PM), calculating a live Environmental Discomfort Index.
* **Knowledge Distillation (KD) & QAT:** The core emotion model (MobileNetV2, alpha=0.5) was trained via Knowledge Distillation and Quantization-Aware Training (INT8), allowing it to run at high FPS purely on the CPU.
* **Time-Series Database:** Logs all metrics (emotion, posture, gesture) to a local InfluxDB instance for historical querying and dashboard visualization.
* **Live SaaS-Style Dashboard:** The AI inference script runs completely decoupled from visualization. A Streamlit web app instantly reads the live data stream, allowing you to view real-time charts from any device on the network.

---

## ⚙️ Requirements & Setup

### Python Dependencies
To run this system, install the required packages in your virtual environment:

```bash
pip install tflite-runtime opencv-python numpy mediapipe
pip install streamlit streamlit-autorefresh plotly pandas influxdb-client
```
*(Note: If running on Raspberry Pi, also install `picamera2`)*

### InfluxDB Server Setup (Required)
The system requires an InfluxDB server to log real-time data.
1. Download and install InfluxDB (v2.x) for your platform (Windows, Linux, or Raspberry Pi OS) from the [InfluxData website](https://docs.influxdata.com/influxdb/v2/install/).
2. Start the InfluxDB service.
3. Access the InfluxDB UI (default `http://localhost:8086`) to complete the initial setup.
4. Create a bucket named `emosys_data`.
5. Generate an API Token and update the `INFLUX_TOKEN`, `INFLUX_ORG`, and `INFLUX_URL` variables in `codes/influxdb_handler.py`.

---

## 🚀 How to Run

EmoSys supports two primary execution modes depending on your hardware.

### Mode A: Raspberry Pi 5 (Pi Camera Module 3)
This mode utilizes the Picamera2 library and includes full support for **emotion**, **body posture**, **hand gesture**, and the **micro climate sensor**.

1. **Start the AI Engine (Terminal 1)**
   ```bash
   cd ~/Downloads/'EmoSys - KD N QAT'/codes
   source ../env/bin/activate
   python qat_student_tflite_pi.py
   ```
2. **Start the Dashboard (Terminal 2)**
   ```bash
   cd ~/Downloads/'EmoSys - KD N QAT'/codes
   source ../env/bin/activate
   streamlit run dashboard.py
   ```

### Mode B: Standard PC / Laptop (Webcam or Still Images)
If you do not have a Raspberry Pi and just want to use a standard USB/integrated webcam or test on still images, run the legacy inference code. 

1. **Start the AI Engine (Terminal 1)**
   ```bash
   cd "codes"
   # To run using your default Webcam:
   python qat_student_tflite.py
   
   # To run a test on a specific still image:
   python qat_student_tflite.py --image "path/to/image.jpg"
   ```
2. **Start the Dashboard (Terminal 2)**
   ```bash
   cd "codes"
   streamlit run dashboard.py
   ```

### View the Analytics
Once Streamlit starts, it will display a local Network URL (e.g., `http://localhost:8501`). Open that URL in your web browser to see the live, real-time analytics dashboard!

---

## 📂 Repository Structure

### Core & Inference
* **`codes/qat_student_tflite_pi.py`**: The main execution engine for Raspberry Pi 5. Captures video via `Picamera2`, runs YuNet, TFLite (Emotion), MediaPipe (Posture/Gesture), and `ClimateReader`.
* **`codes/qat_student_tflite.py`**: The standard execution engine for PC/Webcam and still images.
* **`codes/dashboard.py`**: The Streamlit web application providing a premium SaaS-style UI.
* **`codes/influxdb_handler.py`**: The time-series database handler for persisting metrics.
* **`codes/climate_sensor.py`**: Hardware abstraction layer for I2C climate sensors (Pi only).

### Models
* **`qat_student_int8.tflite`**: The highly compressed INT8 quantized MobileNetV2 emotion model.
* **`face_detection_yunet_2023mar.onnx`**: Extremely lightweight face detection model.
* **`pose_landmarker_lite.task`**: MediaPipe's lightweight pose estimation model.
