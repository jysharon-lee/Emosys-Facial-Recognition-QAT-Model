"""
Retroactively apply shoulder-width normalization to existing CSV data.
The CSVs already have nose-centered coordinates (pts[0] == 0,0,0).
This script divides all coordinates by the shoulder distance (landmarks 11-12)
to make features invariant to body proportions.
"""
import pandas as pd
import numpy as np
import glob, os

DATA_DIR = r"C:\Users\user\Documents\Emosys\EmoSys - KD N QAT\EmoSys - KD N QAT\Gesture Dataset Pi"
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "dataset_gesture_*.csv")))

print(f"Found {len(csv_files)} files to normalize.\n")

for f in csv_files:
    df = pd.read_csv(f)
    feat_cols = [c for c in df.columns if c.startswith('f_')]
    features = df[feat_cols].values.astype(np.float32)

    for i in range(len(features)):
        pts = features[i].reshape(33, 3)
        shoulder_dist = np.linalg.norm(pts[11] - pts[12])
        if shoulder_dist > 1e-4:
            pts /= shoulder_dist
        features[i] = pts.flatten()

    df[feat_cols] = features
    df.to_csv(f, index=False)
    print(f"  Normalized {os.path.basename(f)} ({len(df)} rows)")

print("\nDone! All CSVs now have shoulder-width normalized features.")
