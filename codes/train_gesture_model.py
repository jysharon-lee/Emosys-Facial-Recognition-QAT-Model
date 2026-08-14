import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy.interpolate import interp1d

print("TensorFlow Version:", tf.__version__)

# Configuration
DATASET_DIRS = [
    r"C:\Users\user\Documents\Emosys\EmoSys - KD N QAT\EmoSys - KD N QAT\Gesture Dataset Pi",
]
MODEL_SAVE_PATH = r"C:\Users\user\Documents\Emosys\EmoSys - KD N QAT\EmoSys - KD N QAT\codes\gesture_model.h5"
TFLITE_SAVE_PATH = r"C:\Users\user\Documents\Emosys\EmoSys - KD N QAT\EmoSys - KD N QAT\codes\gesture_model.tflite"

TIME_STEPS = 15
N_CLASSES = 7

GESTURE_LABELS = ["Neutral", "Eye Scratch", "Head Scratch", "Chin Rest",
                  "Nose Scratch", "Neck Rub", "Fidget"]

# ── Distance-based feature engineering ────────────────────────────────────────
# Instead of raw (x,y,z) coordinates, compute distances from each hand to
# key face/head landmarks. These distances are naturally invariant to body
# proportions and person identity.
#
# MediaPipe Pose landmark indices:
#   0=nose, 2=left_eye, 5=right_eye, 7=left_ear, 8=right_ear,
#   9=mouth_left, 10=mouth_right, 11=left_shoulder, 12=right_shoulder,
#   13=left_elbow, 14=right_elbow, 15=left_wrist, 16=right_wrist,
#   19=left_index, 20=right_index

FACE_TARGETS = [0, 2, 5, 7, 8, 9, 10]  # nose, eyes, ears, mouth corners
HAND_LANDMARKS = [15, 16, 19, 20]       # wrists + index fingertips

def angle_between_cosine(a, b, c):
    """Returns the cosine of the angle between vectors ba and bc."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-4 or norm_bc < 1e-4:
        return 0.0
    return np.dot(ba, bc) / (norm_ba * norm_bc)

def engineer_features(pts_flat):
    """
    Convert 99 raw coordinates (33 landmarks x 3, nose-centered + shoulder-normalized)
    into distance-based features that are person-invariant.
    """
    pts = pts_flat.reshape(33, 3)
    feats = []

    neck = (pts[11] + pts[12]) / 2.0  # midpoint of shoulders

    # For each hand point: distances to face targets + neck + vertical position
    for h in HAND_LANDMARKS:
        hand = pts[h]
        for t in FACE_TARGETS:
            feats.append(np.linalg.norm(hand - pts[t]))
        feats.append(np.linalg.norm(hand - neck))
        feats.append(hand[1])  # y-position (height relative to nose)

    # Inter-hand distance (wrists)
    feats.append(np.linalg.norm(pts[15] - pts[16]))

    # Elbow heights (indicates arm raising)
    feats.append(pts[13][1])
    feats.append(pts[14][1])

    # ── Option B: Joint Angle Features ──
    # Left elbow angle (Shoulder[11] - Elbow[13] - Wrist[15])
    feats.append(angle_between_cosine(pts[11], pts[13], pts[15]))
    # Right elbow angle (Shoulder[12] - Elbow[14] - Wrist[16])
    feats.append(angle_between_cosine(pts[12], pts[14], pts[16]))

    return np.array(feats, dtype=np.float32)

# Per hand: 7 face dists + 1 neck dist + 1 y-pos = 9 features
# 4 hands x 9 = 36, + 1 inter-hand + 2 elbows + 2 elbow angles = 41
N_FEATURES = 41

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("\n[1/7] Loading datasets...")

csv_files = []
for d in DATASET_DIRS:
    found = glob.glob(os.path.join(d, "dataset_gesture_*.csv"))
    csv_files.extend(found)
    print(f"  Found {len(found)} files in {os.path.basename(d)}/")

if not csv_files:
    print("ERROR: Cannot find any dataset files.")
    exit()

df_list = []
for file in csv_files:
    print(f"  -> Loading {os.path.basename(file)}")
    df_list.append(pd.read_csv(file))

df = pd.concat(df_list, ignore_index=True)

labels     = df['label'].values
person_ids = df['person_id'].values
raw_features = df.drop(['label', 'person_id'], axis=1).values.astype(np.float32)

unique_persons = sorted(set(person_ids))
print(f"\n  Persons found: {unique_persons}  ({len(unique_persons)} total)")
for pid in unique_persons:
    count = (person_ids == pid).sum()
    print(f"    Person {pid}: {count} frames")

# ── 1.5. Engineer features ────────────────────────────────────────────────────
print(f"\n[1.5/7] Engineering {N_FEATURES} distance-based features from raw coordinates...")
features = np.array([engineer_features(row) for row in raw_features], dtype=np.float32)
print(f"  Transformed {raw_features.shape} -> {features.shape}")

# ── 2. Create Temporal Windows ────────────────────────────────────────────────
print(f"\n[2/7] Creating {TIME_STEPS}-frame time-series sequences...")
X, y, groups = [], [], []

current_seq = []
current_label = -1
current_person = -1

for i in range(len(features)):
    if labels[i] != current_label or person_ids[i] != current_person:
        current_seq = []
        current_label = labels[i]
        current_person = person_ids[i]

    current_seq.append(features[i])

    if len(current_seq) == TIME_STEPS:
        X.append(current_seq)
        y.append(current_label)
        groups.append(current_person)
        current_seq = current_seq[1:]

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
groups = np.array(groups, dtype=np.int32)

print(f"Total sequences generated: {len(X)}")
print(f"Input shape (Samples, Time Steps, Features): {X.shape}")

# ── 3. Leave-One-Person-Out Cross-Validation ──────────────────────────────────
print("\n[3/7] Leave-One-Person-Out Cross-Validation...")
logo = LeaveOneGroupOut()

fold_results = []
for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
    test_person = sorted(set(groups[test_idx]))[0]
    train_persons = sorted(set(groups[train_idx]))

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Data augmentation (noise + temporal warping)
    X_aug, y_aug = list(X_tr), list(y_tr)
    
    def temporal_warp(seq, max_warp=0.15):
        time_ax = np.linspace(0, 1, seq.shape[0])
        f = interp1d(time_ax, seq, axis=0, kind='linear', fill_value='extrapolate')
        warp_factor = np.random.uniform(1.0 - max_warp, 1.0 + max_warp)
        new_time = np.linspace(0, warp_factor, seq.shape[0])
        return f(new_time).astype(np.float32)

    for i in range(len(X_tr)):
        for _ in range(2):  # 2 augmented copies
            # Option A: Temporal Warping (stretch/compress time by ±15%)
            warped = temporal_warp(X_tr[i], max_warp=0.15)
            # Add spatial jitter
            noise = np.random.normal(0, 0.03, warped.shape).astype(np.float32)
            X_aug.append(warped + noise)
            y_aug.append(y_tr[i])
    X_tr = np.array(X_aug, dtype=np.float32)
    y_tr = np.array(y_aug, dtype=np.int32)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(TIME_STEPS, N_FEATURES)),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(N_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_tr, y_tr, epochs=30, batch_size=64, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
              validation_data=(X_te, y_te))

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    fold_results.append((test_person, acc))
    print(f"  Fold {fold+1}: Test person={test_person}, Train persons={train_persons} -> Accuracy: {acc*100:.1f}%")

avg_acc = np.mean([a for _, a in fold_results])
print(f"\n  Average Leave-One-Out Accuracy: {avg_acc*100:.1f}%")

# ── 4. Final model: train on ALL data ─────────────────────────────────────────
print("\n[4/7] Training final model on ALL persons...")

# Augment all data
X_aug, y_aug = list(X), list(y)

def temporal_warp(seq, max_warp=0.15):
    time_ax = np.linspace(0, 1, seq.shape[0])
    f = interp1d(time_ax, seq, axis=0, kind='linear', fill_value='extrapolate')
    warp_factor = np.random.uniform(1.0 - max_warp, 1.0 + max_warp)
    new_time = np.linspace(0, warp_factor, seq.shape[0])
    return f(new_time).astype(np.float32)

for i in range(len(X)):
    for _ in range(2):
        warped = temporal_warp(X[i], max_warp=0.15)
        noise = np.random.normal(0, 0.03, warped.shape).astype(np.float32)
        X_aug.append(warped + noise)
        y_aug.append(y[i])
X_all = np.array(X_aug, dtype=np.float32)
y_all = np.array(y_aug, dtype=np.int32)

print(f"  Training on {len(X_all)} sequences (all persons + augmented)")

final_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(TIME_STEPS, N_FEATURES)),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(N_CLASSES, activation='softmax')
])

final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
final_model.summary()

history = final_model.fit(
    X_all, y_all,
    epochs=30,
    batch_size=64,
    verbose=1
)

final_model.save(MODEL_SAVE_PATH)
print(f"Saved Keras model to: {MODEL_SAVE_PATH}")

# ── 5. Convert to TFLite ─────────────────────────────────────────────────────
print("\n[5/7] Compiling to quantized TensorFlow Lite model for Raspberry Pi...")
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_SAVE_PATH, 'wb') as f:
    f.write(tflite_model)

size_kb = os.path.getsize(TFLITE_SAVE_PATH) / 1024
print(f"Saved TFLite model to: {TFLITE_SAVE_PATH}")
print(f"--> FINAL TFLITE SIZE: {size_kb:.2f} KB <--")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print("\n[6/7] Results Summary:")
print("=" * 60)
for person, acc in fold_results:
    bar = "#" * int(acc * 40)
    print(f"  Person {person}: {acc*100:5.1f}%  {bar}")
print(f"  {'Average':>9s}: {avg_acc*100:5.1f}%")
print("=" * 60)

print("\n--- TRAINING COMPLETE ---")
print(f"Cross-validated accuracy (leave-one-person-out): {avg_acc*100:.1f}%")
print(f"Final model trained on ALL {len(unique_persons)} persons.")
