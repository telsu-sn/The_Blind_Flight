"""
Final inference pipeline
Generates submission.csv
"""

import os, json, time, cv2
import numpy as np
import pandas as pd

from train.terrain_model import terrain_model
from train.desert_tile_model import desert_tile_model
from train.forest_tile_model import forest_tile_model
from train.lab_tile_model import lab_tile_model

from pathfinding.cost_grid import build_cost_grid_general
from pathfinding.start_goal import find_start_goal_safe
from pathfinding.dijkstra import dijkstra

BASE = "SynapseDrive_Dataset"
TEST_IMG = f"{BASE}/test/images"
TEST_VEL = f"{BASE}/test/velocities"

results = []

for fname in sorted(os.listdir(TEST_IMG)):
    img = cv2.imread(os.path.join(TEST_IMG, fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(os.path.join(TEST_VEL, fname.replace(".png",".json"))) as f:
        velocity = np.array(json.load(f)["boost"])

    mean = img.mean(axis=(0,1))
    std  = img.std(axis=(0,1))
    terrain = terrain_model.predict(np.concatenate([mean,std]).reshape(1,-1))[0]

    model = {
        "desert": desert_tile_model,
        "forest": forest_tile_model,
        "lab": lab_tile_model
    }[terrain]

    tile_size = img.shape[0] // 20
    feats = []

    for i in range(20):
        for j in range(20):
            tile = img[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
            mean = tile.mean(axis=(0,1))
            std  = tile.std(axis=(0,1))
            if terrain == "lab":
                gray = tile.mean(axis=2)
                feat = np.concatenate([mean,std,[gray.mean(),gray.std()]])
            else:
                feat = np.concatenate([mean,std])
            feats.append(feat)

    class_grid = model.predict(np.array(feats)).reshape(20,20)
    start, goal = find_start_goal_safe(class_grid, velocity)
    cost_grid = build_cost_grid_general(class_grid, velocity, terrain)
    path,_ = dijkstra(cost_grid, start, goal)

    results.append({
        "image_id": fname.replace(".png",""),
        "path": path or "u"
    })

pd.DataFrame(results).to_csv("submission.csv", index=False)
