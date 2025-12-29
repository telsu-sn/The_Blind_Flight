"""
Cost grid construction
"""

import numpy as np

BASE_COSTS = {
    "desert": {0: 1.2, 1: 1e9, 2: 3.7, 3: 1.2, 4: 2.2},
    "forest": {0: 1.5, 1: 1e9, 2: 2.8, 3: 1.5, 4: 2.5},
    "lab":    {0: 1.0, 1: 1e9, 2: 3.0, 3: 1.0, 4: 2.0}
}

def build_cost_grid_general(class_grid, velocity_grid, terrain):
    cost = np.zeros_like(velocity_grid, dtype=float)
    for cls, base_cost in BASE_COSTS[terrain].items():
        cost[class_grid == cls] = base_cost
    cost -= velocity_grid
    return cost
