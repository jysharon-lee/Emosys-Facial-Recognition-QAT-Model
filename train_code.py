!pip install -q --upgrade numpy
!pip install -q --upgrade tensorflow tensorflow-model-optimization
import importlib, pkg_resources
importlib.reload(pkg_resources)

# --------


# ══════════════════════════════════════════════════════════════════════
# USAGE NOTES
# ══════════════════════════════════════════════════════════════════════
#
# Kaggle notebook setup cell (run before this script):
# -------------------------------------------------------
# !pip install tensorflow-model-optimization -q
#
# Expected dataset structure (adjust paths in CONFIG section above):
# -------------------------------------------------------
# /kaggle/input/fer2013/
#   train/
#     angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
#   val/
#     angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
#
# /kaggle/input/raf-db/
#   train/  val/   (same class layout)
#
# All 4 phase .keras files + final_qat_int8.tflite land in:
#   /kaggle/working/qat_phases/
#
# TensorBoard (run in a separate cell):
# -------------------------------------------------------
# %load_ext tensorboard
# %tensorboard --logdir /kaggle/working/qat_phases/tb_logs
#
# P100 memory notes:
# -------------------------------------------------------
# • BATCH_SIZE=32 at 224×224 uses ~6 GB GPU RAM with QAT overhead.
# • Set BATCH_SIZE=16 if you add heavy augmentation or hit OOM.
# • Mixed precision (MIXED_PREC=True) cuts memory ~30% on P100.
# • ds.cache() is used; make sure /kaggle/working has enough disk.
#   If RAM is tight, remove .cache() from build_dataset().
# 
# ══════════════════════════════════════════════════════════════════════

# --------

# ── stdlib ─────────────────────────────────────────────────────────────
import os, gc, time, warnings
warnings.filterwarnings("ignore")

# --------

# ── third-party ────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

print(f"TensorFlow  : {tf.__version__}")
print(f"tfmot       : {tfmot.__version__}")
print(f"GPU devices : {tf.config.list_physical_devices('GPU')}")

# --------

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# --------

# ── Paths ──────────────────────────────────────────────────────────────
FER13_TRAIN  = "/kaggle/input/datasets/msambare/fer2013/train"
FER13_VAL    = "/kaggle/input/datasets/msambare/fer2013/test"
RAFDB_TRAIN  = "/kaggle/input/datasets/shuvoalok/raf-db-dataset/DATASET/train"
RAFDB_VAL    = "/kaggle/input/datasets/shuvoalok/raf-db-dataset/DATASET/test"
AFD_TRAIN    = "/kaggle/input/datasets/qystyy/asian-emotion-faces-split/dataset_split/train"
AFD_VAL      = "/kaggle/input/datasets/qystyy/asian-emotion-faces-split/dataset_split/test"
OUTPUT_DIR   = "/kaggle/working/qat_phases"

# ── Model ──────────────────────────────────────────────────────────────
NUM_CLASSES  = 7          # anger disgust fear happy neutral sad surprise
IMAGE_SIZE   = (224, 224)
CHANNELS     = 3

# ── Training ───────────────────────────────────────────────────────────
BATCH_SIZE   = 32         # safe for P100 16 GB at 224×224
MIXED_PREC   = True       # float16 activations → 2× throughput on P100

# ── Phase schedule ─────────────────────────────────────────────────────
#   (name, freeze_spec, dataset, lr, max_epochs, early_stop_patience)
#   freeze_spec: "ALL" | int N  (freeze layers 0..N-1)
PHASES = [                                                  # 1st Trial - This became the teacher model 
    ("phase1_all_frozen", "ALL",   "fer13",  7e-4, 15, 4),  # All the setting is the same as before
    ("phase2_top90_frozen",  90,   "fer13",  7e-4, 15, 4),  # 
    ("phase3_top40_frozen",  40,   "fer13",  3e-6, 30, 5),  # 
    ("phase4_rafdb",         40,   "rafdb",  3e-6, 75, 5),  # 
    ("phase5_afd",           40,   "afd",    5e-6, 120, 5), # 
]
os.makedirs(OUTPUT_DIR, exist_ok=True)
print('Config OK - output dir:', OUTPUT_DIR)

# --------


def augment_train(image, label):
    """
    Light augmentation — enough to regularise without destroying subtle
    facial expressions.  Keep it mild because QAT already introduces
    extra noise via fake-quant ops.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.12)
    image = tf.image.random_contrast(image, lower=0.88, upper=1.12)
    # Mild random crop: zoom in up to 10 %
    shape  = tf.shape(image)
    crop_h = tf.cast(tf.cast(shape[0], tf.float32) * 0.92, tf.int32)
    crop_w = tf.cast(tf.cast(shape[1], tf.float32) * 0.92, tf.int32)
    image  = tf.image.resize(image, IMAGE_SIZE)
    image  = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

print("Data aug done")

# --------

# FER-13 alphabetical order — Keras default, no remapping needed
FER13_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# RAF-DB: folder name → correct emotion index (matching FER-13 order above)
# RAF-DB numbering: 1=Surprise,2=Fear,3=Disgust,4=Happy,5=Sad,6=Angry,7=Neutral
RAFDB_FOLDER_TO_EMOTION = {
    '1': 6,   # Surprise → index 6
    '2': 2,   # Fear     → index 2
    '3': 1,   # Disgust  → index 1
    '4': 3,   # Happy    → index 3
    '5': 5,   # Sad      → index 5
    '6': 0,   # Angry    → index 0
    '7': 4,   # Neutral  → index 4
}


def build_dataset(directory, training, class_mapping=None):
    """
    class_mapping: dict {original_index: correct_index}
                   None = use Keras alphabetical order as-is (FER-13)
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=training,
        seed=42,
        interpolation='bilinear',
    )

    # Remap labels if a mapping is provided
    if class_mapping is not None:
        # Build a lookup tensor: index i → correct index
        n_classes  = NUM_CLASSES
        lookup     = np.zeros(n_classes, dtype=np.int32)
        for orig, correct in class_mapping.items():
            lookup[int(orig)] = correct
        lookup_tensor = tf.constant(lookup, dtype=tf.int32)

        def remap_labels(images, labels):
            # labels shape: (batch, n_classes) one-hot
            orig_indices  = tf.argmax(labels, axis=1, output_type=tf.int32)
            new_indices   = tf.gather(lookup_tensor, orig_indices)
            new_labels    = tf.one_hot(new_indices, depth=n_classes)
            return images, new_labels

        ds = ds.map(remap_labels, num_parallel_calls=tf.data.AUTOTUNE)

    # Normalise
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment_train, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_datasets(name):
    if name == 'fer13':
        return build_dataset(FER13_TRAIN, True), build_dataset(FER13_VAL, False)
    elif name == 'rafdb':
        # Build index mapping: Keras assigns 0..6 alphabetically to '1'..'7'
        # so original index = folder_name - 1
        index_mapping = {i: RAFDB_FOLDER_TO_EMOTION[str(i+1)] for i in range(7)}
        return (build_dataset(RAFDB_TRAIN, True,  index_mapping),
                build_dataset(RAFDB_VAL,   False, index_mapping))
    elif name == 'afd':
        return build_dataset(AFD_TRAIN, True), build_dataset(AFD_VAL, False)
    elif name == 'vea':
        return build_dataset(VEA_TRAIN, True), build_dataset(VEA_VAL, False)
    raise ValueError(f'Unknown dataset: {name}')

print('Dataset pipeline ready.')

# --------

# Sanity check — print a few label distributions per dataset
for name in ['fer13', 'rafdb','afd']:
    train_ds, _ = get_datasets(name)
    label_counts = np.zeros(NUM_CLASSES, dtype=int)
    for _, labels in train_ds.take(20):
        indices = np.argmax(labels.numpy(), axis=1)
        for idx in indices:
            label_counts[idx] += 1
    print(f'\n{name} label sample (first 20 batches):')
    for i, (emotion, count) in enumerate(zip(FER13_CLASSES, label_counts)):
        print(f'  {i} {emotion:<10} : {count}')

# --------

# 1. Build base model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, CHANNELS),
    include_top=False,
    weights='imagenet', #defualt alpha = 1.0
)

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)  # (None, 1280)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation='softmax',
    dtype='float32',
    name='predictions',
)(x)

model = tf.keras.Model(base_model.input, outputs=output, name='MobileNetV2_QAT')

# 2. Save the clean base model BEFORE quantizing
base_save_path = os.path.join(OUTPUT_DIR, 'base_model.keras')
model.save(base_save_path)
print('Base model saved.')

print(f'Total layers : {len(model.layers)}')
print(f'Total params : {model.count_params():,}')
model.summary()

# --------

# ══════════════════════════════════════════════════════════════════════
# 5.  QAT WRAPPER  (RUN ONCE ONLY)
# ══════════════════════════════════════════════════════════════════════
print('Inserting fake-quantisation nodes ...')
model = tfmot.quantization.keras.quantize_model(model)
print(f'QAT model — layers after wrapping: {len(model.layers)}')
print('Fake-quant nodes inserted. QAT wrapper will be carried through all phases.')

# --------

def apply_freeze(model, freeze_spec):
    """
    freeze_spec = 'ALL'  → freeze every layer
    freeze_spec = int N  → freeze layers 0..N-1, unfreeze N..end
    Must recompile after calling this.
    """
    if freeze_spec == 'ALL':
        for layer in model.layers:
            layer.trainable = False
        n_frozen, n_trainable = len(model.layers), 0
    else:
        n = int(freeze_spec)
        for i, layer in enumerate(model.layers):
            layer.trainable = (i >= n)
        n_frozen    = n
        n_trainable = len(model.layers) - n

    trainable_params = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    print(f'  Frozen / trainable layers : {n_frozen} / {n_trainable}')
    print(f'  Trainable parameters      : {trainable_params:,}')
    return model

# --------

def build_callbacks(phase_name, patience, lr):
    best_ckpt = os.path.join(OUTPUT_DIR, f'{phase_name}_best.keras')

    if os.path.exists(best_ckpt):
        os.remove(best_ckpt)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_ckpt,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,   # full model preserves QAT graph
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            min_delta=0.0005,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=lr * 1e-3,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(OUTPUT_DIR, f'{phase_name}_log.csv'),
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(OUTPUT_DIR, 'tb_logs', phase_name),
            update_freq='epoch',
        ),
        EpochSummary(),
    ], best_ckpt


# --------

class EpochSummary(tf.keras.callbacks.Callback):
    """One-line per-epoch print so verbose=0 still shows progress."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr   = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        print(
            f'  ep {epoch+1:03d} | '
            f'loss={logs.get("loss", 0):.4f} | '
            f'acc={logs.get("accuracy", 0):.4f} | '
            f'val_loss={logs.get("val_loss", 0):.4f} | '
            f'val_acc={logs.get("val_accuracy", 0):.4f} | '
            f'lr={lr:.2e}'
        )

# --------

def compute_class_weights(dataset_name):
    """
    Auto-compute class weights from the training directory.
    Returns a dict {class_index: weight} ready for model.fit(class_weight=...)
    """
    if dataset_name == 'fer13':
        train_dir = FER13_TRAIN
    elif dataset_name == 'rafdb':
        train_dir = RAFDB_TRAIN
    elif dataset_name == 'afd':
        train_dir = AFD_TRAIN
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    # Count samples per class by scanning the folder
    class_names = sorted(os.listdir(train_dir))   # alphabetical — matches Keras label order
    class_counts = []
    for cls in class_names:
        cls_path = os.path.join(train_dir, cls)
        if os.path.isdir(cls_path):
            class_counts.append(len(os.listdir(cls_path)))

    total_samples = sum(class_counts)
    labels_expanded = []
    for idx, count in enumerate(class_counts):
        labels_expanded.extend([idx] * count)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(class_names)),
        y=np.array(labels_expanded),
    )

    class_weight_dict = dict(enumerate(weights))

    # Print breakdown so you can see what weights were assigned
    print(f'\n[Class Weights] Dataset: {dataset_name}  |  Total samples: {total_samples}')
    print(f'  {"Class":<12} {"Count":>7}  {"Weight":>8}')
    print(f'  {"-"*30}')
    for idx, (cls, count, w) in enumerate(zip(class_names, class_counts, weights)):
        bar = '█' * int(w * 6)
        print(f'  {cls:<12} {count:>7}  {w:>8.4f}  {bar}')

    return class_weight_dict

# --------

phase_name, freeze_spec, dataset, lr, max_epochs, patience = PHASES[0]
print(f'=== {phase_name.upper()} ===')

# ✅ Always rebuild QAT model fresh each phase to avoid stale HDF5 state
with tfmot.quantization.keras.quantize_scope():
    model = tf.keras.models.load_model(base_save_path)

model = apply_freeze(model, freeze_spec)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-7),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision')],
)

train_ds, val_ds = get_datasets(dataset)
callbacks_p1, best_ckpt_p1 = build_callbacks(phase_name, patience, lr)
cw = compute_class_weights(dataset)

t0 = time.time()
history_p1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=max_epochs, callbacks=callbacks_p1, class_weight=cw,
)

print(f'\nPhase 1 done in {(time.time()-t0)/60:.1f} min')
print(f'Best val_acc : {max(history_p1.history["val_accuracy"]):.4f}')
print(f'Best val_precision : {max(history_p1.history["val_precision"]):.4f}')

# Reload best checkpoint so next phase starts from the best weights
with tfmot.quantization.keras.quantize_scope():
    model = tf.keras.models.load_model(base_save_path)
model.load_weights(best_ckpt_p1)

model.save(os.path.join(OUTPUT_DIR, f'{phase_name}_final_accuracy_{max(history_p1.history["val_accuracy"]):.4f}.keras'))
print(f'Saved → {phase_name}_final.keras')

del train_ds, val_ds
gc.collect()

# --------

phase_name, freeze_spec, dataset, lr, max_epochs, patience = PHASES[1]
print(f'=== {phase_name.upper()} ===')

model = apply_freeze(model, freeze_spec)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-7),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy',tf.keras.metrics.Precision(name='precision')],
)

train_ds, val_ds = get_datasets(dataset)
callbacks_p2, best_ckpt_p2 = build_callbacks(phase_name, patience, lr)
cw = compute_class_weights(dataset)

t0 = time.time()
history_p2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=max_epochs, callbacks=callbacks_p2, class_weight=cw,
)
print(f'\nPhase 2 done in {(time.time()-t0)/60:.1f} min')
print(f'Best val_acc : {max(history_p2.history["val_accuracy"]):.4f}')
print(f'Best val_precision : {max(history_p2.history["val_precision"]):.4f}')

with tfmot.quantization.keras.quantize_scope():
    model = tf.keras.models.load_model(base_save_path)
model.load_weights(best_ckpt_p2)

model.save(os.path.join(OUTPUT_DIR, f'{phase_name}_final_accuracy_{max(history_p2.history["val_accuracy"]):.4f}.keras'))
print(f'Saved → {phase_name}_final.keras')

del train_ds, val_ds
gc.collect()

# --------

phase_name, freeze_spec, dataset, lr, max_epochs, patience = PHASES[2]
print(f'=== {phase_name.upper()} ===')

model = apply_freeze(model, freeze_spec)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-7),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy',tf.keras.metrics.Precision(name='precision')],
)

train_ds, val_ds = get_datasets(dataset)
callbacks_p3, best_ckpt_p3 = build_callbacks(phase_name, patience, lr)
cw = compute_class_weights(dataset)

t0 = time.time()
history_p3 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=max_epochs, callbacks=callbacks_p3, class_weight=cw,
)
print(f'\nPhase 3 done in {(time.time()-t0)/60:.1f} min')
print(f'Best val_acc : {max(history_p3.history["val_accuracy"]):.4f}')
print(f'Best val_precision : {max(history_p3.history["val_precision"]):.4f}')

with tfmot.quantization.keras.quantize_scope():
    model = tf.keras.models.load_model(base_save_path)
model.load_weights(best_ckpt_p3)

model.save(os.path.join(OUTPUT_DIR, f'{phase_name}_final_accuracy_{max(history_p3.history["val_accuracy"]):.4f}.keras'))
print(f'Saved → {phase_name}_final.keras')

del train_ds, val_ds
gc.collect()

# --------


phase_name, freeze_spec, dataset, lr, max_epochs, patience = PHASES[3]
print(f'=== {phase_name.upper()} ===')

model = apply_freeze(model, freeze_spec)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-7),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy',tf.keras.metrics.Precision(name='precision')],
)

train_ds, val_ds = get_datasets(dataset)
callbacks_p4, best_ckpt_p4 = build_callbacks(phase_name, patience, lr)
cw = compute_class_weights(dataset)

t0 = time.time()
history_p4 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=max_epochs, callbacks=callbacks_p4, class_weight=cw
)
print(f'\nPhase 4 done in {(time.time()-t0)/60:.1f} min')
print(f'Best val_acc : {max(history_p4.history["val_accuracy"]):.4f}')
print(f'Best val_precision : {max(history_p4.history["val_precision"]):.4f}')

with tfmot.quantization.keras.quantize_scope():
    model = tf.keras.models.load_model(base_save_path) 
model.load_weights(best_ckpt_p4)

model.save(os.path.join(OUTPUT_DIR, f'{phase_name}_final_accuracy_{max(history_p4.history["val_accuracy"]):.4f}.keras'))
print(f'Saved → {phase_name}_final.keras')

del train_ds, val_ds
gc.collect()


# --------

phase_name, freeze_spec, dataset, lr, max_epochs, patience = PHASES[4]
print(f'=== {phase_name.upper()} ===')

model = apply_freeze(model, freeze_spec)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-7),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy',tf.keras.metrics.Precision(name='precision')],
)

train_ds, val_ds = get_datasets(dataset)
callbacks_p5, best_ckpt_p5 = build_callbacks(phase_name, patience, lr)
cw = compute_class_weights(dataset)

t0 = time.time()
history_p5 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=max_epochs, callbacks=callbacks_p5, class_weight=cw,
)
print(f'\nPhase 5 done in {(time.time()-t0)/60:.1f} min')
print(f'Best val_acc : {max(history_p5.history["val_accuracy"]):.4f}')
print(f'Best val_precision : {max(history_p5.history["val_precision"]):.4f}')

with tfmot.quantization.keras.quantize_scope():
    model = tf.keras.models.load_model(base_save_path)
model.load_weights(best_ckpt_p5)

model.save(os.path.join(OUTPUT_DIR, f'{phase_name}_final_accuracy_{max(history_p5.history["val_accuracy"]):.4f}.keras'))
print(f'Saved → {phase_name}_final.keras')

del train_ds, val_ds
gc.collect()

# --------

def build_fitnet_student(num_classes=NUM_CLASSES, input_shape=(*IMAGE_SIZE, CHANNELS)):
    student_base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=0.5,               
        include_top=False,
        weights='imagenet',      
        pooling=None  
    )
    
    student_features = student_base.output 
    
    x = tf.keras.layers.GlobalAveragePooling2D()(student_features)
    x = tf.keras.layers.Dense(128)(x)         
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)          
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, name='student_logits')(x)
    
    flat_student = tf.keras.Model(
        inputs=student_base.input, 
        outputs=[student_features, outputs], 
        name='emosys_student_fitnet'
    )
    return flat_student

student = build_fitnet_student()
print("Student FitNet constructed. Total params:", f"{student.count_params():,}")

# --------

import glob
import os
import tensorflow as tf

print("Loading Teacher for Feature Distillation...")

# Dynamically search for the Phase 5 teacher model, whatever its accuracy number is
search_pattern = os.path.join(OUTPUT_DIR, 'phase5_afd_final_accuracy_*.keras')
teacher_matches = glob.glob(search_pattern)

# Sort and grab the most recently generated file
teacher_matches.sort()
teacher_path = teacher_matches[-1]
print(f"Found Teacher Model: {os.path.basename(teacher_path)}")

# Load the model
teacher_original = tf.keras.models.load_model(teacher_path)

# Extract the feature layer
teacher_feature_layer = teacher_original.get_layer('out_relu')

# Create the multi-output FitNet wrapper
teacher_fitnet = tf.keras.Model(
    inputs=teacher_original.input,
    outputs=[teacher_feature_layer.output, teacher_original.output],
    name='teacher_fitnet'
)
teacher_fitnet.trainable = False

# Alias it for the training loop
teacher = teacher_fitnet 
print("Teacher FitNet ready. Outputs: Features and Logits.")

# --------

class EmosysFitNetDistiller(tf.keras.Model):
    def __init__(self, student, teacher, temperature=4.0, alpha=0.3, beta=50.0):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha     
        self.beta = beta       
        
        # 1x1 Conv Adapter: Projects student feature dimensions to perfectly match the teacher
        self.feature_adapter = tf.keras.layers.Conv2D(1280, (1, 1), name='fitnet_adapter')

    def compile(self, optimizer, student_loss_fn, distillation_loss_fn, feature_loss_fn, metrics=None):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.feature_loss_fn = feature_loss_fn 

    def train_step(self, data):
        x, y = data[:2]
        sample_weight = data[2] if len(data) == 3 else None

        t_features, teacher_out = self.teacher(x, training=False)
        
        row_sums = tf.reduce_sum(teacher_out, axis=-1, keepdims=True)
        is_probs = tf.reduce_all(tf.abs(row_sums - 1.0) < 0.01)
        teacher_logits = tf.cond(
            is_probs,
            lambda: tf.math.log(tf.clip_by_value(teacher_out, 1e-6, 1.0 - 1e-6)),
            lambda: teacher_out
        )

        with tf.GradientTape() as tape:
            s_features, student_logits = self.student(x, training=True)
            s_features_adapted = self.feature_adapter(s_features)

            student_loss = self.student_loss_fn(y, student_logits, sample_weight=sample_weight)

            soft_teacher = tf.nn.softmax(teacher_logits / self.temperature)
            soft_student_log = tf.nn.log_softmax(student_logits / self.temperature)
            distillation_loss = tf.reduce_mean(
                tf.reduce_sum(soft_teacher * (tf.math.log(tf.clip_by_value(soft_teacher, 1e-8, 1.0)) - soft_student_log), axis=-1)
            ) * (self.temperature ** 2)
            
            feature_loss = self.feature_loss_fn(t_features, s_features_adapted)

            total_loss = (self.alpha * student_loss) + ((1.0 - self.alpha) * distillation_loss) + (self.beta * feature_loss)

        trainable_vars = self.student.trainable_variables + self.feature_adapter.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        student_probs = tf.nn.softmax(student_logits)
        self.compiled_metrics.update_state(y, student_probs)

        results = {m.name: m.result() for m in self.metrics}
        results.update({
            's_loss': student_loss,
            'kd_loss': distillation_loss,
            'feat_loss': feature_loss,
            'total_loss': total_loss,
        })
        return results

    def test_step(self, data):
        x, y = data[:2]
        _, student_logits = self.student(x, training=False)
        student_probs  = tf.nn.softmax(student_logits)
        self.compiled_metrics.update_state(y, student_probs)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x):
        _, logits = self.student(x)
        return tf.nn.softmax(logits)

# Instantiate the new FitNet Distiller
distiller = EmosysFitNetDistiller(
    student=student, 
    teacher=teacher_fitnet, 
    temperature=4.0, 
    alpha=0.3,
    beta=50.0  
)
print("FitNet Distiller ready")

# --------

import gc, time

# FIXED phase schedule — alpha ramps DOWN as training stabilises
# Phase 1-2: rely heavily on hard labels (student head is random)
# Phase 3-5: trust teacher more once student has a foothold
KD_PHASES = [
    ("kd_phase1_all_frozen",    "ALL", "fer13",  7e-4, 15,  4, 0.9),  # 90% hard labels
    ("kd_phase2_top60_frozen",   60,   "fer13",  3e-5, 15,  4, 0.8),  # 80% hard labels
    ("kd_phase3_top30_frozen",   30,   "fer13",  3e-6, 30,  5, 0.6),  # 60% hard labels
    ("kd_phase4_rafdb",          30,   "rafdb",  3e-6, 75,  5, 0.5),  # balanced
    ("kd_phase5_afd",            30,   "afd",    5e-6, 120, 5, 0.4),  # 40% hard, let teacher guide
]

print("Starting Knowledge Distillation")
student_histories = []

for phase_idx, (phase_name, freeze_spec, dataset_name, lr, max_epochs, patience, alpha) in enumerate(KD_PHASES):
    print(f"\n{'='*60}")
    print(f">>> STUDENT KD PHASE {phase_idx+1}: {phase_name.upper()} <<<")
    print(f"{'='*60}")

   # Since the model is flat, the backbone is everything EXCEPT the last 5 layers we added
    backbone_layers = student.layers[:-5]
    head_layers = student.layers[-5:]

    # 1. ALWAYS make sure the new classification head is trainable
    for layer in head_layers:
        layer.trainable = True

    # 2. Apply freezing logic strictly to the backbone
    if freeze_spec == 'ALL':
        for layer in backbone_layers:
            layer.trainable = False
        print(f"  Backbone: fully frozen ({len(backbone_layers)} layers)")
    else:
        n = int(freeze_spec)
        for i, layer in enumerate(backbone_layers):
            layer.trainable = (i >= n)
        
        n_trainable = sum(1 for l in backbone_layers if l.trainable)
        print(f"  Backbone: frozen 0–{n-1}, trainable {n}–end ({n_trainable} trainable layers)")

  
    distiller.alpha = alpha  # update dynamically per phase
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-7),
        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1
        ),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),  # kept for API compat
        feature_loss_fn=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision')]
    )

    # ── Datasets and class weights ─────────────────────────────────────
    train_ds, val_ds = get_datasets(dataset_name)
    cw = compute_class_weights(dataset_name)

    # ── Checkpoint path for this KD phase ─────────────────────────────
    kd_ckpt_path = os.path.join(OUTPUT_DIR, f'{phase_name}_best.keras')
    if os.path.exists(kd_ckpt_path):
        os.remove(kd_ckpt_path)

    phase_callbacks = [
        # ModelCheckpoint saves the full Distiller — we extract student weights after
        tf.keras.callbacks.ModelCheckpoint(
            filepath=kd_ckpt_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,   # weights only — avoids saving Distiller graph
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='total_loss',
            patience=patience,
            restore_best_weights=True,
            min_delta=0.0005,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='total_loss',
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=lr * 1e-3,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(OUTPUT_DIR, f'{phase_name}_log.csv')
        ),
        EpochSummary(),   # your existing callback — works unchanged on Distiller
    ]

    sample_batch = next(iter(train_ds.take(1)))
    x_sample = sample_batch[0][:4]
   # Unpack the list from the FitNet
    t_features, t_logits = teacher_fitnet(x_sample, training=False)

    # Check ranges separately
    print("Feature map range:", tf.reduce_min(t_features).numpy(), "to", tf.reduce_max(t_features).numpy())
    print("Logits range:", tf.reduce_min(t_logits).numpy(), "to", tf.reduce_max(t_logits).numpy())

    t0 = time.time()
    history = distiller.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        class_weight=cw,
        callbacks=phase_callbacks,
    )
    elapsed = (time.time() - t0) / 60

    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n{phase_name} done in {elapsed:.1f} min")
    print(f"Best val_accuracy  : {best_val_acc:.4f}")
    print(f"Best val_precision : {max(history.history['val_precision']):.4f}")

    # Save student model (not the Distiller wrapper) for inspection
    student_phase_path = os.path.join(
        OUTPUT_DIR,
        f'{phase_name}_student_acc_{best_val_acc:.4f}.keras'
    )
    student.save(student_phase_path)
    print(f"Student saved → {student_phase_path}")

    student_histories.append((phase_name, history))
    del train_ds, val_ds
    gc.collect()

print("\nAll Student Distillation phases finished.")
print(f"Final student params: {student.count_params():,}")

# --------

import tensorflow_model_optimization as tfmot
import os
import glob
import tensorflow as tf

print("Applying strictly QAT to distilled student ...")

# 1. Dynamically find the perfectly distilled student from Phase 5
search_pattern = os.path.join(OUTPUT_DIR, 'kd_phase5_afd_student_acc_*.keras')
student_matches = glob.glob(search_pattern)

if not student_matches:
    raise FileNotFoundError(f"Could not find the Phase 5 student model in {OUTPUT_DIR}")

student_matches.sort()
saved_student_path = student_matches[-1]
print(f"Found Student Model: {os.path.basename(saved_student_path)}")

fitnet_student = tf.keras.models.load_model(saved_student_path)

# Convert dual-output FitNet back to a standard single-output model
student_classification_only = tf.keras.Model(
    inputs=fitnet_student.input, 
    outputs=fitnet_student.get_layer('student_logits').output,
    name='student_qat_ready'
)

# 2. Wrap the model with fake-quant nodes (The core QAT step)
student_qat = tfmot.quantization.keras.quantize_model(student_classification_only)

# 3. CRITICAL FIX: Freeze all Batch Normalization layers to prevent collapse on tiny dataset
for layer in student_qat.layers:
    # TFMOT wraps layers, so we have to check the inner layer type
    if hasattr(layer, 'layer') and isinstance(layer.layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    elif isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# 4. Compile with enough learning rate for the model to "heal"
student_qat.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-7),  
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05),  
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision')]
)

# 5. QAT Calibration using the target domain (AFD)
train_ds_afd, val_ds_afd = get_datasets('afd')

print("\nStarting QAT Recovery Pass...")
student_qat.fit(
    train_ds_afd,
    validation_data=val_ds_afd,
    epochs=15,  # Give it enough time to map the new 8-bit buckets
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=4, restore_best_weights=True
        ),
        EpochSummary() 
    ]
)

# Save QAT student — this becomes the model for TFLite export
student_qat_path = os.path.join(OUTPUT_DIR, 'student_qat_final.keras')
student_qat.save(student_qat_path)
print(f"\nQAT student saved → {student_qat_path}")

# --------

all_histories = [
    ('Phase 1\nFER-13 7e-4', history_p1),
    ('Phase 2\nFER-13 7e-4', history_p2),
    ('Phase 3\nFER-13 3e-6', history_p3),
    ('Phase 4\nRAF-DB 3e-6', history_p4),
    ('Phase 5\nAFD-DB 5e-6', history_p5),
]

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle('MobileNetV2 KD + QAT — Teacher Training Curves per Phase', fontsize=13, y=1.01)

for col, (title, hist) in enumerate(all_histories):
    epochs = range(1, len(hist.history['accuracy']) + 1)

    # Accuracy
    axes[0, col].plot(epochs, hist.history['accuracy'],     label='train', marker='o', ms=3)
    axes[0, col].plot(epochs, hist.history['val_accuracy'], label='val',   marker='s', ms=3, linestyle='--')
    axes[0, col].set_title(title, fontsize=10)
    axes[0, col].set_ylabel('Accuracy' if col == 0 else '')
    axes[0, col].set_ylim(0, 1)
    axes[0, col].legend(fontsize=8)
    axes[0, col].grid(True, alpha=0.3)

    # Loss
    axes[1, col].plot(epochs, hist.history['loss'],     label='train', marker='o', ms=3)
    axes[1, col].plot(epochs, hist.history['val_loss'], label='val',   marker='s', ms=3, linestyle='--')
    axes[1, col].set_ylabel('Loss' if col == 0 else '')
    axes[1, col].set_xlabel('Epoch')
    axes[1, col].legend(fontsize=8)
    axes[1, col].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'kd_teacher_training_curves.png')
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.show()
print(f'Plot saved → {plot_path}')

# --------

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle('MobileNetV2 KD + QAT — Student Training Curves per Phase', fontsize=13, y=1.01)

for col, (title, hist) in enumerate(student_histories):
    epochs = range(1, len(hist.history['accuracy']) + 1)

    # ── Accuracy ──
    axes[0, col].plot(epochs, hist.history['accuracy'],     label='train', marker='o', ms=3)
    axes[0, col].plot(epochs, hist.history['val_accuracy'], label='val',   marker='s', ms=3, linestyle='--')
    axes[0, col].set_title(title, fontsize=10)
    axes[0, col].set_ylabel('Accuracy' if col == 0 else '')
    axes[0, col].set_ylim(0, 1)
    axes[0, col].legend(fontsize=8)
    axes[0, col].grid(True, alpha=0.3)

    # ── Loss ──
    # Safely fetch 'total_loss' since custom distiller doesn't use standard 'loss'
    train_loss = hist.history.get('total_loss', [])
    if train_loss:
        axes[1, col].plot(epochs, train_loss, label='total_loss (train)', marker='o', ms=3)
    
    # Safely plot val_loss only if it exists
    if 'val_loss' in hist.history:
        axes[1, col].plot(epochs, hist.history['val_loss'], label='val_loss', marker='s', ms=3, linestyle='--')
        
    axes[1, col].set_ylabel('Loss' if col == 0 else '')
    axes[1, col].set_xlabel('Epoch')
    axes[1, col].legend(fontsize=8)
    axes[1, col].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'kd_student_training_curves.png')
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.show()
print(f'Plot saved → {plot_path}')

# --------

import glob

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Helper function to dynamically find the correct saved model file
def find_model_path(phase_prefix):
    search_pattern = os.path.join(OUTPUT_DIR, f'{phase_prefix}_final_accuracy_*.keras')
    matches = glob.glob(search_pattern)
    if not matches:
        # Fallback just in case it saved differently
        fallback_pattern = os.path.join(OUTPUT_DIR, f'{phase_prefix}_final.keras')
        matches = glob.glob(fallback_pattern)
        if not matches:
            raise FileNotFoundError(f"Could not find model for {phase_prefix} in {OUTPUT_DIR}")
    return matches[0]

# Now we automatically fetch the correct filenames!
phase_models = [
    ('Phase 1\nFER-13 7e-4',  find_model_path('phase1_all_frozen'),   'fer13'),
    ('Phase 2\nFER-13 7e-4',  find_model_path('phase2_top90_frozen'), 'fer13'),
    ('Phase 3\nFER-13 3e-6',  find_model_path('phase3_top40_frozen'), 'fer13'),
    ('Phase 4\nRAF-DB 3e-6',  find_model_path('phase4_rafdb'),        'rafdb'),
    ('Phase 5\nAFD-DB 5e-6',  find_model_path('phase5_afd'),          'afd'),
]

fig, axes = plt.subplots(1, 5, figsize=(28, 5))
fig.suptitle('MobileNetV2 KD + QAT — Teacher Confusion Matrix per Phase (% of true class)', fontsize=13, y=1.02)

for col, (title, ckpt_path, dataset_name) in enumerate(phase_models):
    print(f"Evaluating {title.replace(chr(10), ' ')} using {os.path.basename(ckpt_path)}...")

    # ── Load phase model ──────────────────────────────────────────────
    with tfmot.quantization.keras.quantize_scope():
        phase_model = tf.keras.models.load_model(ckpt_path)

    # ── Get val dataset ───────────────────────────────────────────────
    _, val_ds = get_datasets(dataset_name)

    # ── Collect true labels + predictions ────────────────────────────
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = phase_model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    # ── Confusion matrix normalised to % of true class (row-wise) ────
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = (cm.astype(float) / cm.sum(axis=1, keepdims=True)) * 100

    # ── Plot ──────────────────────────────────────────────────────────
    sns.heatmap(
        cm_pct,
        ax=axes[col],
        annot=True,
        fmt='.1f',            # one decimal place e.g. 87.3
        cmap='Blues',
        vmin=0, vmax=100,
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        linewidths=0.4,
        cbar=col == 4,        # only show colorbar on last plot
        annot_kws={'size': 7},
    )
    axes[col].set_title(title, fontsize=10)
    axes[col].set_xlabel('Predicted', fontsize=8)
    axes[col].set_ylabel('True' if col == 0 else '', fontsize=8)
    axes[col].tick_params(axis='x', rotation=45, labelsize=7)
    axes[col].tick_params(axis='y', rotation=0,  labelsize=7)

    # ── Print per-phase summary ───────────────────────────────────────
    overall_acc = np.diag(cm).sum() / cm.sum() * 100
    print(f'Overall acc: {overall_acc:.1f}%\n')

    del phase_model
    gc.collect()

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'teacher_confusion_matrices.png')
plt.savefig(cm_path, dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved → {cm_path}')

# --------

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import gc

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def find_student_model(phase_prefix):
    search_pattern = os.path.join(OUTPUT_DIR, f'{phase_prefix}_student_acc_*.keras')
    matches = glob.glob(search_pattern)
    if not matches:
        raise FileNotFoundError(f"Could not find model for {phase_prefix} in {OUTPUT_DIR}")
    
    matches.sort()
    return matches[-1]

# Exactly matching the 5 KD phases of the teacher model
student_models = [
    ('Phase 1 (KD)\nFER-13', find_student_model('kd_phase1_all_frozen'),   'fer13'),
    ('Phase 2 (KD)\nFER-13', find_student_model('kd_phase2_top60_frozen'), 'fer13'),
    ('Phase 3 (KD)\nFER-13', find_student_model('kd_phase3_top30_frozen'), 'fer13'),
    ('Phase 4 (KD)\nRAF-DB', find_student_model('kd_phase4_rafdb'),        'rafdb'),
    ('Phase 5 (KD)\nAFD-DB', find_student_model('kd_phase5_afd'),          'afd'),
]

fig, axes = plt.subplots(1, 5, figsize=(28, 5))
fig.suptitle('MobileNetV2 KD — Student Confusion Matrix Progression (% of true class)', fontsize=13, y=1.02)

for col, (title, ckpt_path, dataset_name) in enumerate(student_models):
    print(f"Evaluating {title.replace(chr(10), ' ')} using {os.path.basename(ckpt_path)}...")

    # Load the normal KD model 
    with tfmot.quantization.keras.quantize_scope():
        phase_model = tf.keras.models.load_model(ckpt_path, compile=False)

    _, val_ds = get_datasets(dataset_name)

    y_true, y_pred = [], []
    for images, labels in val_ds:
        # CRITICAL FIX: Extract only the logits from the FitNet dual-output
        preds = phase_model.predict(images, verbose=0)
        student_logits = preds[1]
        
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(student_logits, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    cm_pct = (cm.astype(float) / cm.sum(axis=1, keepdims=True)) * 100

    sns.heatmap(
        cm_pct, ax=axes[col], annot=True, fmt='.1f', cmap='Greens',
        vmin=0, vmax=100, xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
        linewidths=0.4, cbar=(col == 4), annot_kws={'size': 7},
    )
    axes[col].set_title(title, fontsize=10)
    axes[col].set_xlabel('Predicted', fontsize=8)
    axes[col].set_ylabel('True' if col == 0 else '', fontsize=8)
    axes[col].tick_params(axis='x', rotation=45, labelsize=7)
    axes[col].tick_params(axis='y', rotation=0,  labelsize=7)

    overall_acc = np.diag(cm).sum() / cm.sum() * 100
    print(f'Overall acc: {overall_acc:.1f}%\n')

    del phase_model
    gc.collect()

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'student_kd_confusion_matrices.png')
plt.savefig(cm_path, dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved → {cm_path}')

# --------

print('Converting QAT model → TFLite INT8 ...')

converter = tf.lite.TFLiteConverter.from_keras_model(student_qat)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# QAT already calibrated activation ranges — no representative
# dataset needed for INT8 conversion
tflite_bytes = converter.convert()

tflite_path = os.path.join(OUTPUT_DIR, 'student_int8.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_bytes)

print(f'Saved → {tflite_path}')
print(f'File size : {len(tflite_bytes)/1024:.1f} KB')
print(f'Expected  : 900–1100 KB  (target: <1500 KB)')

# Quick dtype sanity check
interpreter_check = tf.lite.Interpreter(model_content=tflite_bytes)
interpreter_check.allocate_tensors()
inp = interpreter_check.get_input_details()[0]
out = interpreter_check.get_output_details()[0]
print(f'\nInput  dtype : {inp["dtype"]}  shape: {inp["shape"]}')
print(f'Output dtype : {out["dtype"]}  shape: {out["shape"]}')
print("Note: dtype may be float32 even on QAT export without explicit INT8 ops override.")
print("If you need strict INT8 I/O, see the INT8-only block below.")

# --------

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

inp_detail  = interpreter.get_input_details()[0]
out_detail  = interpreter.get_output_details()[0]
dummy_input = np.random.rand(1, *IMAGE_SIZE, CHANNELS).astype(np.float32)

interpreter.set_tensor(inp_detail['index'], dummy_input)
interpreter.invoke()
preds = interpreter.get_tensor(out_detail['index'])

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(f'Input  shape : {inp_detail["shape"]}  dtype: {inp_detail["dtype"]}')
print(f'Output shape : {out_detail["shape"]}  dtype: {out_detail["dtype"]}')
print(f'Predicted class : {EMOTION_LABELS[np.argmax(preds)]} ({preds.max():.3f})')
print('TFLite INT8 inference OK ✓')

# --------

def evaluate_tflite(tflite_path, dataset_name):
    """
    Run full evaluation on the TFLite INT8 model.
    Reports accuracy, precision, recall, F1 per class + overall.
    """

    # ── Load TFLite model ─────────────────────────────────────────────
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    # ── Run inference on full val set ─────────────────────────────────
    _, val_ds = get_datasets(dataset_name)

    y_true, y_pred, y_prob = [], [], []
    for images, labels in val_ds:
        for i in range(len(images)):
            img = images[i].numpy()[np.newaxis].astype(np.float32)  # (1, 224, 224, 3)
            interpreter.set_tensor(inp_detail['index'], img)
            interpreter.invoke()
            prob = interpreter.get_tensor(out_detail['index'])[0]   # (7,)
            y_true.append(np.argmax(labels[i].numpy()))
            y_pred.append(np.argmax(prob))
            y_prob.append(prob)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # ── Overall metrics ───────────────────────────────────────────────
    overall_acc = (y_true == y_pred).mean() * 100
    print(f'\n{"═"*55}')
    print(f'  TFLite INT8 Evaluation — {dataset_name.upper()} val set')
    print(f'{"═"*55}')
    print(f'  Overall Accuracy : {overall_acc:.2f}%')

    # ── Per-class report ──────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=EMOTION_LABELS,
        digits=4,
        output_dict=True,
    )

    # Print formatted table
    print(f'\n  {"Class":<12} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Support":>10}')
    print(f'  {"─"*52}')
    for label in EMOTION_LABELS:
        r = report[label]
        print(
            f'  {label:<12}'
            f'  {r["precision"]*100:>8.2f}%'
            f'  {r["recall"]*100:>8.2f}%'
            f'  {r["f1-score"]*100:>8.2f}%'
            f'  {int(r["support"]):>9}'
        )
    print(f'  {"─"*52}')
    macro = report['macro avg']
    print(
        f'  {"Macro avg":<12}'
        f'  {macro["precision"]*100:>8.2f}%'
        f'  {macro["recall"]*100:>8.2f}%'
        f'  {macro["f1-score"]*100:>8.2f}%'
    )
    weighted = report['weighted avg']
    print(
        f'  {"Weighted avg":<12}'
        f'  {weighted["precision"]*100:>8.2f}%'
        f'  {weighted["recall"]*100:>8.2f}%'
        f'  {weighted["f1-score"]*100:>8.2f}%'
    )

    # ── Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = (cm.astype(float) / cm.sum(axis=1, keepdims=True)) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_pct, ax=ax,
        annot=True, fmt='.1f', cmap='Blues',
        vmin=0, vmax=100,
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        linewidths=0.4,
        annot_kws={'size': 9},
    )
    ax.set_title(f'TFLite INT8 — Confusion Matrix (% of true class)\n{dataset_name.upper()} val set', fontsize=11)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, f'{dataset_name}_tflite_confusion_matrix.png')
    plt.savefig(cm_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'\nConfusion matrix saved → {cm_path}')

    return report


# ── Run it ────────────────────────────────────────────────────────────
tflite_report = evaluate_tflite(tflite_path, dataset_name='fer13')
tflite_report = evaluate_tflite(tflite_path, dataset_name='rafdb')
tflite_report = evaluate_tflite(tflite_path, dataset_name='afd')

# --------

import shutil

# Zips the entire output directory into one file
shutil.make_archive("/kaggle/working/emosys_final_export", 'zip', OUTPUT_DIR)

print("All models and plots safely zipped into emosys_final_export.zip")
print("Download this zip file from the Kaggle output panel")

# --------

#import shutil

# put in the path of the new exported model folder
#shutil.make_archive("/kaggle/working/phase5_afd_final_accuracy_0.8988", 'zip', "/kaggle/working/phase5_afd_final_accuracy_0.8988")

# --------

import os

print('\n── Output files ───────────────────────────────────────')
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.isfile(fpath):
        kb = os.path.getsize(fpath) / 1024
        print(f'  {fname:<45s} {kb:>8.0f} KB')

print("\n--- Student KD Phases ---")
for phase_name, history in student_histories:
    best_acc       = max(history.history.get('val_accuracy', [0.0]))
    
    # FitNet distiller drops val_loss to save compute time, so we safely fetch total_loss
    loss_list      = history.history.get('val_loss', history.history.get('total_loss', [0.0]))
    best_loss      = min(loss_list)
    
    n_ep           = len(history.history.get('val_accuracy', [0]))
    best_precision = max(history.history.get('val_precision', [0.0]))
    print(f'{phase_name:35s}  epochs={n_ep:2d}  val_acc={best_acc:.4f}  min_loss={best_loss:.4f}   val_precision={best_precision:.4f}')
    
try:
    print("\n--- Teacher Phases ---")
    for label, hist in [
        ('Phase 1 (FER-13 7e-4)', history_p1),
        ('Phase 2 (FER-13 7e-4)', history_p2),
        ('Phase 3 (FER-13 3e-6)', history_p3),
        ('Phase 4 (RAF-DB 3e-6)', history_p4),
        ('Phase 5 (AFD-DB 5e-6)', history_p5),
    ]:
        best_acc       = max(hist.history['val_accuracy'])
        best_loss      = min(hist.history['val_loss'])
        n_ep           = len(hist.history['val_accuracy'])
        best_precision = max(hist.history['val_precision'])
        print(f'{label:30s}  epochs={n_ep:2d}  val_acc={best_acc:.4f}  val_loss={best_loss:.4f}   val_precision={best_precision:.4f}')
except NameError:
    print("\n--- Teacher Phases ---")
    print("Teacher phases were loaded from disk, skipping training history summary.")

# --------

