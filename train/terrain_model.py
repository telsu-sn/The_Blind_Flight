"""
Terrain classification model.
Classifies an image as one of: desert, forest, lab
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------

BASE = Path("SynapseDrive_Dataset")
TRAIN_IMG = BASE / "train" / "images"
TRAIN_LBL = BASE / "train" / "labels"

X_terrain, y_terrain = [], []

for fname in sorted(os.listdir(TRAIN_IMG)):
    if not fname.endswith(".png"):
        continue

    img_path = TRAIN_IMG / fname
    lbl_path = TRAIN_LBL / fname.replace(".png", ".json")

    if not lbl_path.exists():
        continue

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(lbl_path, "r") as f:
        data = json.load(f)

    terrain = data.get("terrain")
    if terrain is None:
        continue

    # Global RGB statistics
    mean_rgb = img.mean(axis=(0, 1))
    std_rgb  = img.std(axis=(0, 1))
    feat = np.concatenate([mean_rgb, std_rgb])

    X_terrain.append(feat)
    y_terrain.append(terrain)

X_terrain = np.array(X_terrain)
y_terrain = np.array(y_terrain)

# -------------------------------------------------
# MODEL
# -------------------------------------------------

terrain_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance"))
])

scores = cross_val_score(terrain_model, X_terrain, y_terrain, cv=3)
terrain_model.fit(X_terrain, y_terrain)

print("Terrain model trained")
print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
