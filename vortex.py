import numpy as np
import math


class VortexEffect:
    """
    Gesture 2 — Right fist.

    Pulls all particles into a flat spiral orbiting the centre.
    Each particle has a precomputed target radius and phase offset so the
    spiral looks natural rather than all particles collapsing to the same ring.
    The orbit tightens while the gesture is held and relaxes on release.
    """

    def __init__(self, particles):
        self.particles = particles
        self.strength  = 0.0          # 0 → 1 formation blend
        self.spin      = 0.0          # accumulated rotation offset

        n = len(particles.positions)

        # Precompute per-particle spiral targets (stable, no per-frame randoms)
        self.radii  = np.random.uniform(30, 200, n)    # how far from centre
        self.phases = np.random.uniform(0, 2 * math.pi, n)  # angle offset
        self.heights = np.random.normal(0, 15, n)      # slight Z spread

    def update(self, active, time):
        if active:
            self.strength = min(1.0, self.strength + 0.04)
        else:
            self.strength = max(0.0, self.strength - 0.04)

        if self.strength == 0:
            return

        # Spin rate scales with formation strength
        self.spin += 0.03 * (1.0 + self.strength)

        # Vectorised: compute target positions for the whole particle array
        angles = self.phases + self.spin + self.radii * 0.005  # inner orbits faster

        targets = np.stack([
            self.radii  * np.cos(angles),
            self.radii  * np.sin(angles),
            self.heights * (1.0 - self.strength * 0.6),   # flatten as strength grows
        ], axis=1)                                          # (N, 3)

        # Lerp toward targets weighted by formation strength
        self.particles.positions += (
            targets - self.particles.positions
        ) * (0.06 * self.strength)
