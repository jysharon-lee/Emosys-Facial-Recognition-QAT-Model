# EmoSys - Real-Time Edge AI Emotion Detection

EmoSys is a lightweight, highly optimized Edge AI emotion recognition system. It utilizes a custom-trained **MobileNetV2** architecture to detect 7 core human emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) in real-time via a standard USB webcam. 

This project was specifically engineered for deployment on constrained edge devices (e.g., Raspberry Pi, low-end PCs) by aggressive model compression and calibration techniques.

## Key Features

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
3. Run the live webcam inference script:

```bash
python qat_student_tflite.py
```

4. A window will open displaying your webcam feed with real-time bounding boxes and a live emotion confidence graph. 
5. To exit the program, press **`q`** on your keyboard.

## Calibration (Optional)

If the model behaves slightly differently due to your specific room lighting or face shape, you can easily tune the model's sensitivity without retraining it.
Open `qat_student_tflite.py` and modify the `calibration_biases` array to apply positive or negative numerical boosts to specific emotions.
