import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("TensorFlow Version:", tf.__version__)

# Configuration
DATASET_DIR = r"C:\Users\user\Documents\Emosys\EmoSys - KD N QAT\EmoSys - KD N QAT\Gesture Dataset"
MODEL_SAVE_PATH = r"C:\Users\user\Documents\Emosys\EmoSys - KD N QAT\EmoSys - KD N QAT\codes\gesture_model.h5"
TFLITE_SAVE_PATH = r"C:\Users\user\Documents\Emosys\EmoSys - KD N QAT\EmoSys - KD N QAT\codes\gesture_model.tflite"

TIME_STEPS = 15  # Window of 15 frames (~0.5 seconds of movement)
N_FEATURES = 99  # 33 pose landmarks * 3 coordinates
N_CLASSES = 7

# 1. Load Data
print("\n[1/7] Loading datasets...")
import glob
csv_files = glob.glob(os.path.join(DATASET_DIR, "dataset_gesture_*.csv"))

if not csv_files:
    print(f"ERROR: Cannot find any dataset files in {DATASET_DIR}")
    exit()

df_list = []
for file in csv_files:
    print(f"  -> Loading {os.path.basename(file)}")
    df_list.append(pd.read_csv(file))

df = pd.concat(df_list, ignore_index=True)

labels = df['label'].values
features = df.drop('label', axis=1).values

# 2. Create Temporal Windows (Sequences)
print(f"\n[2/7] Creating {TIME_STEPS}-frame time-series sequences...")
X, y = [], []

# We create sequences but ensure we don't mix different labels in a single sequence
current_seq = []
current_label = -1

for i in range(len(features)):
    if labels[i] != current_label:
        current_seq = []
        current_label = labels[i]
        
    current_seq.append(features[i])
    
    if len(current_seq) == TIME_STEPS:
        X.append(current_seq)
        y.append(current_label)
        # Shift sequence by 1 frame (overlap) for maximum data extraction
        current_seq = current_seq[1:]

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print(f"Total sequences generated: {len(X)}")
print(f"Input shape (Samples, Time Steps, Features): {X.shape}")

# 3. Train/Test Split
print("\n[3/7] Splitting data into Training and Validation sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training on {len(X_train)} sequences, Validating on {len(X_test)} sequences.")

# 4. Build LSTM Model
print("\n[4/7] Building Keras LSTM Model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(TIME_STEPS, N_FEATURES)),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(N_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 5. Train Model
print("\n[5/7] Training Model...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    callbacks=callbacks
)

# 6. Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Validation Accuracy: {acc*100:.2f}%")

model.save(MODEL_SAVE_PATH)
print(f"Saved unoptimized Keras model to: {MODEL_SAVE_PATH}")

# 7. Convert to TFLite (Quantization)
print("\n[6/7] Compiling to quantized TensorFlow Lite model for Raspberry Pi...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_SAVE_PATH, 'wb') as f:
    f.write(tflite_model)
    
size_kb = os.path.getsize(TFLITE_SAVE_PATH) / 1024
print(f"Saved TFLite model to: {TFLITE_SAVE_PATH}")
print(f"--> FINAL TFLITE SIZE: {size_kb:.2f} KB <--")

# 8. Plot Confusion Matrix
print("\n[7/7] Generating Confusion Matrix Plot...")
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Neutral", "Eye Scratch", "Head Scratch", "Chin Rest", "Nose Scratch", "Neck Rub", "Fidget"],
            yticklabels=["Neutral", "Eye Scratch", "Head Scratch", "Chin Rest", "Nose Scratch", "Neck Rub", "Fidget"])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title(f'Gesture Model Validation (Accuracy: {acc*100:.2f}%)')
plt.tight_layout()
cm_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "confusion_matrix.png")
plt.savefig(cm_path)
print(f"Saved confusion matrix plot to: {cm_path}")

print("\n--- PHASE 2 COMPLETE ---")
print("You are ready to move to Phase 3 (Raspberry Pi integration).")
