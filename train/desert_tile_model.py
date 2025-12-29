"""
Tile classification model for DESERT terrain
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def extract_tile_features(img, grid, tile_size):
    X, y = [], []
    for i in range(20):
        for j in range(20):
            tile = img[
                i*tile_size:(i+1)*tile_size,
                j*tile_size:(j+1)*tile_size
            ]
            mean = tile.mean(axis=(0,1))
            std  = tile.std(axis=(0,1))
            X.append(np.concatenate([mean, std]))
            y.append(grid[i, j])
    return np.array(X), np.array(y)

desert_tile_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="euclidean"
    ))
])
