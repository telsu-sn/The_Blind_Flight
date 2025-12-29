"""
Dijkstra path planning on 20x20 grid
"""

import heapq

def dijkstra(cost_grid, start, goal):
    moves = {'u':(-1,0), 'd':(1,0), 'l':(0,-1), 'r':(0,1)}
    pq = [(0, start, [])]
    visited = set()

    while pq:
        cost, (x,y), path = heapq.heappop(pq)
        if (x,y) in visited:
            continue
        visited.add((x,y))

        if (x,y) == goal:
            return ''.join(path), cost

        for m,(dx,dy) in moves.items():
            nx, ny = x+dx, y+dy
            if 0 <= nx < 20 and 0 <= ny < 20:
                if cost_grid[nx,ny] < 1e8:
                    heapq.heappush(
                        pq,
                        (cost + cost_grid[nx,ny], (nx,ny), path + [m])
                    )

    return "", float("inf")
