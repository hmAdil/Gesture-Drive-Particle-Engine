import numpy as np
from config import PARTICLES, BOUNDARY


class ParticleSystem:
    def __init__(self):
        self.positions = np.random.uniform(-300, 300, (PARTICLES, 3))
        self.velocities = np.random.uniform(-0.3, 0.3, (PARTICLES, 3))

    def drift(self):
        self.positions += self.velocities
        self.velocities *= 0.985

        # Vectorized boundary clamp — replaces the O(n) Python loop
        norms = np.linalg.norm(self.positions, axis=1)          # (N,)
        over = norms > BOUNDARY                                  # bool mask
        if np.any(over):
            scale = BOUNDARY / norms[over, np.newaxis]           # broadcast-safe
            self.positions[over] = self.positions[over] * scale
            self.velocities[over] *= -0.5

    def burst(self):
        # Fully vectorized random burst — replaces the O(n) Python loop
        directions = np.random.uniform(-1, 1, (len(self.positions), 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions /= norms + 1e-6

        speeds = np.random.uniform(6.0, 10.0, (len(self.positions), 1))
        self.velocities[:] = directions * speeds
