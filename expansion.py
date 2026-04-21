import numpy as np
from config import BOUNDARY


class ExpansionEffect:
    """
    Gesture 4 — Both open palms.

    While held, gently pushes every particle outward toward the boundary,
    spreading the cloud to fill the full 3D volume. Feels like the simulation
    is exhaling or expanding on command.

    On release the idle drift in ParticleSystem takes over naturally — no
    special collapse logic needed since particles already have boundary
    reflection baked into drift().
    """

    PUSH_STRENGTH = 0.12   # outward nudge per frame (scales with distance from boundary)

    def __init__(self, particles):
        self.particles = particles
        self.strength  = 0.0

    def update(self, active):
        if active:
            self.strength = min(1.0, self.strength + 0.03)
        else:
            self.strength = max(0.0, self.strength - 0.03)

        if self.strength == 0:
            return

        positions  = self.particles.positions          # (N, 3)
        norms      = np.linalg.norm(positions, axis=1, keepdims=True)   # (N, 1)

        # Avoid pushing particles that are already at the boundary
        near_boundary = (norms[:, 0] / BOUNDARY)      # 0 → 1 distance fraction
        push_scale    = (1.0 - near_boundary)          # stronger push toward centre
        push_scale    = np.clip(push_scale, 0, 1)

        # Outward unit direction
        safe_norms   = np.where(norms < 1e-6, 1.0, norms)
        directions   = positions / safe_norms           # (N, 3)

        self.particles.velocities += (
            directions
            * push_scale[:, np.newaxis]
            * self.PUSH_STRENGTH
            * self.strength
        )
