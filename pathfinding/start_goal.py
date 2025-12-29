"""
Safe start and goal detection
"""

import numpy as np

def find_start_goal_safe(class_grid, velocity_grid):
    start_pos = np.argwhere(class_grid == 3)
    goal_pos  = np.argwhere(class_grid == 4)
    walkable  = np.argwhere(class_grid != 1)

    if len(start_pos):
        start = tuple(start_pos[0])
    else:
        idx = np.argmax(velocity_grid[walkable[:,0], walkable[:,1]])
        start = tuple(walkable[idx])

    if len(goal_pos):
        goal = tuple(goal_pos[0])
    else:
        idx = np.argmin(velocity_grid[walkable[:,0], walkable[:,1]])
        goal = tuple(walkable[idx])

    return start, goal
