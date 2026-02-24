import numpy as np
import random
from config import PARTICLES, BOUNDARY

class ParticleSystem:
    def __init__(self):
        self.positions = np.random.uniform(-300, 300, (PARTICLES, 3))
        self.velocities = np.random.uniform(-0.3, 0.3, (PARTICLES, 3))

    def drift(self):
        self.positions += self.velocities
        self.velocities *= 0.985

        for i in range(len(self.positions)):
            if np.linalg.norm(self.positions[i]) > BOUNDARY:
                self.positions[i] = self.positions[i] / np.linalg.norm(self.positions[i]) * BOUNDARY
                self.velocities[i] *= -0.5

    def burst(self):
        for i in range(len(self.positions)):
            direction = np.random.uniform(-1, 1, 3)
            direction /= (np.linalg.norm(direction) + 0.001)

            speed = random.uniform(6.0, 10.0)
            self.velocities[i] = direction * speed