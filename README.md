# EmoSys - Real-Time Edge AI Emotion Detection
### This project was built throughout my internship at SMD Semiconductor

EmoSys is a lightweight, highly optimized Edge AI emotion recognition system. It utilizes a custom-trained **MobileNetV2** architecture to detect 7 core human emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) in real-time via a standard USB webcam. 

This project was specifically engineered for deployment on constrained edge devices (e.g., Raspberry Pi, low-end PCs) by aggressive model compression and calibration techniques.

## Key Features

* **Multi-Face Tracking:** Features a built-in `CentroidTracker` capable of simultaneously tracking individual faces with persistent Face IDs. Each person receives their own isolated temporal smoothing buffer and separate emotion history graphs/logs.
* **Knowledge Distillation (KD):** A small, fast "Student" model (MobileNetV2, alpha=0.5) was trained by mimicking the deep feature extraction of a massive "Teacher" model, retaining high accuracy while cutting inference time by 80%.
* **Quantization-Aware Training (QAT):** The model weights are quantized down to **INT8 (8-bit)** integers, severely reducing RAM usage and allowing for high FPS on edge CPUs without suffering from catastrophic mode collapse.
* **Focal Loss Implementation:** Forces the lightweight network to focus on difficult micro-expressions (like furrowed brows) rather than relying on lazy macro-features (like open mouths), completely eliminating "Sad" and "Surprise" biases.
* **Live Logit Calibration:** Uses mathematical Prior Adjustment during inference to dynamically bridge the "Domain Gap" between brightly-lit studio training datasets and real-world webcam environments.

## Requirements

To run the live inference script, you will need a standard webcam and the following Python packages installed:

```bash
pip install tensorflow opencv-python numpy
```

## How to Run

1. Clone or download this repository.
2. Ensure you have the `qat_student_int8.tflite` model file in the same directory as the script.
3. Run the inference script:

**Live Webcam Mode:**
```bash
python qat_student_tflite.py
```

**Static Image Injection Mode:**
To eliminate live webcam noise and test the model's raw inference capabilities on a single sterile frame, use the `--image` flag:
```bash
python qat_student_tflite.py --image sad.jpeg
```

4. A window will open displaying the feed/image with real-time bounding boxes, individual Face IDs, and live emotion confidence predictions. 
5. To exit the program, press **`q`** on your keyboard. Upon exiting, the system will generate and save a per-face summary graph containing each tracked individual's emotion timelines.

## Calibration (Optional)

If the model behaves slightly differently due to your specific room lighting or face shape, you can easily tune the model's sensitivity without retraining it.
Open `qat_student_tflite.py` and modify the `calibration_biases` array to apply positive or negative numerical boosts to specific emotions.

## Repository Structure
### Core Models & Inference
* **`qat_student_tflite.py`**: The main execution script. It captures the live webcam feed (or a static image), utilizes YuNet for rapid face detection, and runs the quantized student model for real-time, multi-face emotion tracking. Handles live drawing, smoothing, and end-of-session graph generation.
* **`qat_student_int8.tflite`**: The highly compressed INT8 quantized MobileNetV2 "Student" model used for edge inference.
* **`face_detection_yunet_2023mar.onnx`**: The YuNet face detection model. It is extremely lightweight and is used by the system to locate face bounding boxes before passing them to the emotion classifier.

### Training & Development
* **`v2-t2-kd-qat-model-mobilenetv2.ipynb`** (and `.py`): The complete training pipeline. Contains the code for Knowledge Distillation (Teacher-to-Student feature matching), Focal Loss implementation to balance difficult classes, Quantization-Aware Training (QAT), and exporting the final `.tflite` model.
* **`labels_FER.json`**: A simple dictionary file mapping the model's numerical tensor outputs to the 7 human emotion string labels (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise).
* **`requirement.txt`**: Lists all required Python dependencies to run the inference and training environments.

### Output & Logging Directories
* **`graph/`**: Stores the automatically generated per-face summary charts (line charts for emotion over time + bar charts for average confidence) saved when you exit the script.
* **`log/`**: Stores timestamped text files containing raw emotion confidence data and inference speed diagnostics for post-session programmatic analysis.
* **`run_counter/`**: A small utility folder that keeps track of the total number of trial sessions run by the system.
* **`Still_Images/`**: Contains static reference photos used to run the inference script in `--image` injection mode, providing a sterile baseline without webcam noise.
