import numpy as np


class ShockwaveEffect:
    """
    Gesture 3 — Left open palm (edge-triggered on raise).

    On each new gesture detection, fires a single radial shockwave:
    every particle gets a velocity impulse directed away from the origin,
    proportional to its proximity (closer = stronger kick). The wave
    propagates naturally through the existing physics in ParticleSystem.drift().

    This effect does NOT hold state across frames — it fires once per edge
    and then the particle system's own momentum carries the result.
    """

    # Impulse strength at the epicentre; falls off with distance
    BASE_IMPULSE  = 18.0
    FALLOFF       = 0.003   # impulse = BASE / (1 + FALLOFF * distance)

    def __init__(self, particles):
        self.particles = particles

    def trigger(self):
        """Call this once on the rising edge of the left_open gesture."""
        positions = self.particles.positions           # (N, 3)

        # Direction: away from origin
        directions = positions.copy()
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms < 1e-6, 1.0, norms)   # avoid div-by-zero at origin
        directions /= norms

        # Falloff: particles near the core get a bigger kick
        distances = norms[:, 0]                        # (N,)
        impulse = self.BASE_IMPULSE / (1.0 + self.FALLOFF * distances)

        self.particles.velocities += directions * impulse[:, np.newaxis]
