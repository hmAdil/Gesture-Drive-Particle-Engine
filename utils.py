import numpy as np
import math
from config import WIDTH, HEIGHT, FOV


def project(point):
    x, y, z = point
    factor = FOV / (FOV + z)
    return int(x * factor + WIDTH / 2), int(y * factor + HEIGHT / 2)


def rotate_x(point, angle):
    x, y, z = point
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([x, y * c - z * s, y * s + z * c])


# ── Batch versions for use in the render loop ────────────────────────────────

def rotate_x_batch(points, angle):
    """
    Rotate an (N, 3) array around the X axis.
    Replaces the per-particle rotate_x call in main.py's render loop.
    """
    c = math.cos(angle)
    s = math.sin(angle)
    result = np.empty_like(points)
    result[:, 0] = points[:, 0]
    result[:, 1] = points[:, 1] * c - points[:, 2] * s
    result[:, 2] = points[:, 1] * s + points[:, 2] * c
    return result


def project_batch(points):
    """
    Project an (N, 3) array to (N, 2) screen coordinates.
    Returns integer pixel arrays ready for rendering.
    """
    z = points[:, 2]
    factor = FOV / (FOV + z)                           # (N,)
    px = (points[:, 0] * factor + WIDTH  / 2).astype(np.int32)
    py = (points[:, 1] * factor + HEIGHT / 2).astype(np.int32)
    return px, py
