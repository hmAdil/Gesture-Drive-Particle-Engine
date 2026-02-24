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