"""
Tile classification model for LAB terrain
Uses RGB + grayscale statistics
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

lab_tile_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=3,
        weights="distance",
        metric="euclidean"
    ))
])
