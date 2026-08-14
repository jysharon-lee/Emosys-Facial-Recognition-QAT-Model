import os
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d

print("TensorFlow Version:", tf.__version__)

# ── Configuration ─────────────────────────────────────────────────────────────
curr_dir = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_PATH = os.path.join(curr_dir, "gesture_model.h5")
CALIB_CSV_PATH = os.path.join(curr_dir, "calibration_data.csv")
SAVE_H5_PATH = os.path.join(curr_dir, "gesture_model_personal.h5")
SAVE_TFLITE_PATH = os.path.join(curr_dir, "gesture_model_personal.tflite")

TIME_STEPS = 15

if not os.path.exists(CALIB_CSV_PATH):
    print(f"ERROR: Cannot find calibration data at {CALIB_CSV_PATH}")
    print("Please run `calibrate_user.py` first.")
    exit(1)

if not os.path.exists(BASE_MODEL_PATH):
    print(f"ERROR: Cannot find base model at {BASE_MODEL_PATH}")
    exit(1)

# ── 1. Load Calibration Data ──────────────────────────────────────────────────
print("\n[1/4] Loading Calibration Data...")
df = pd.read_csv(CALIB_CSV_PATH)
labels = df['label'].values
features = df.drop(['label', 'person_id'], axis=1).values.astype(np.float32)

print(f"Loaded {len(df)} frames of personalized data.")

# ── 2. Create Sequences ───────────────────────────────────────────────────────
print(f"\n[2/4] Creating {TIME_STEPS}-frame sequences...")
X_base, y_base = [], []
current_seq = []
current_label = -1

for i in range(len(features)):
    if labels[i] != current_label:
        current_seq = []
        current_label = labels[i]
        
    current_seq.append(features[i])
    
    if len(current_seq) == TIME_STEPS:
        X_base.append(current_seq)
        y_base.append(current_label)
        current_seq = current_seq[1:]  # Sliding window

X_base = np.array(X_base, dtype=np.float32)
y_base = np.array(y_base, dtype=np.int32)
print(f"Generated {len(X_base)} sequences.")

# ── 3. Data Augmentation ──────────────────────────────────────────────────────
print("\n[3/4] Augmenting calibration data to prevent overfitting...")
def temporal_warp(seq, max_warp=0.15):
    time_ax = np.linspace(0, 1, seq.shape[0])
    f = interp1d(time_ax, seq, axis=0, kind='linear', fill_value='extrapolate')
    warp_factor = np.random.uniform(1.0 - max_warp, 1.0 + max_warp)
    new_time = np.linspace(0, warp_factor, seq.shape[0])
    return f(new_time).astype(np.float32)

X_train, y_train = list(X_base), list(y_base)
for i in range(len(X_base)):
    for _ in range(4):  # Create 4 variations per sequence
        warped = temporal_warp(X_base[i], max_warp=0.15)
        noise = np.random.normal(0, 0.03, warped.shape).astype(np.float32)
        X_train.append(warped + noise)
        y_train.append(y_base[i])

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)

# Shuffle training data
indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

print(f"Augmented dataset size: {len(X_train)} sequences.")

# ── 4. Fine-Tune Model ────────────────────────────────────────────────────────
print("\n[4/4] Fine-tuning the base model...")
base_model = tf.keras.models.load_model(BASE_MODEL_PATH)

# Freeze convolutional layers (feature extractors)
# Let the model keep its understanding of human skeletons, but retrain the Dense decision layers
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.Conv1D):
        layer.trainable = False

# Recompile model with frozen layers and a smaller learning rate
base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                   loss='sparse_categorical_crossentropy', 
                   metrics=['accuracy'])

base_model.summary()

# Train for a few epochs
base_model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    verbose=1
)

# ── 5. Save Personalized Models ───────────────────────────────────────────────
base_model.save(SAVE_H5_PATH)
print(f"\nSaved personalized Keras model to: {SAVE_H5_PATH}")

print("Compiling to personalized TFLite model...")
converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(SAVE_TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)
    
print(f"Saved personalized TFLite model to: {SAVE_TFLITE_PATH}")
print("\n--- CALIBRATION COMPLETE ---")
print("Your personalized model is ready! Next time you run qat_student_tflite_pi.py, it will load this profile.")
