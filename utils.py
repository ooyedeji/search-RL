import numpy as np


def flood_orthogonal(image: np.ndarray, seed_point: tuple[int]):
    assert image.ndim == len(seed_point)
    seed_x, seed_y = seed_point
    rows, cols = image.shape
    seed_value = image[seed_x, seed_y]

    filled_image = np.zeros_like(image)
    queue = [(seed_x, seed_y)]
    visited = set()

    while queue:
        x, y = queue.pop(0)

        if (x, y) in visited:
            continue

        filled_image[x, y] = 1
        visited.add((x, y))

        # Expand frontier to neighbors (no diagonals)
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        for i, j in neighbors:
            if 0 <= i < rows and 0 <= j < cols and image[i, j] == seed_value:
                queue.append((i, j))

    return filled_image
