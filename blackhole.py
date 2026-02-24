import numpy as np
import math
import random


class BlackHoleEffect:
    def __init__(self, particles):
        self.particles = particles
        self.formation = 0.0
        self.rotation_speed = 1.0

        self.sphere_indices = np.random.choice(
            len(particles.positions),
            len(particles.positions) // 3,
            replace=False
        )
        self.disk_indices = np.setdiff1d(
            np.arange(len(particles.positions)),
            self.sphere_indices
        )

        self.disk_radii = np.random.uniform(140, 240, len(self.disk_indices))
        self.disk_angles = np.random.uniform(0, 2 * math.pi, len(self.disk_indices))
        self.disk_heights = np.random.normal(0, 8, len(self.disk_indices))

        self.sphere_theta = np.random.uniform(0, 2 * math.pi, len(self.sphere_indices))
        self.sphere_phi = np.random.uniform(0, math.pi, len(self.sphere_indices))
        self.sphere_radius = 90

        self.jet_particles = []

    def update(self, forming, jet_mode, time):

        if forming:
            self.formation = min(1.0, self.formation + 0.05)
        else:
            self.formation = max(0.0, self.formation - 0.05)

        if self.formation == 0:
            self.jet_particles.clear()
            return

        if jet_mode:
            self.rotation_speed = min(2.0, self.rotation_speed + 0.05)
        else:
            self.rotation_speed = max(1.0, self.rotation_speed - 0.05)

        for idx_i, p_index in enumerate(self.sphere_indices):
            theta = self.sphere_theta[idx_i]
            phi = self.sphere_phi[idx_i]

            target = np.array([
                self.sphere_radius * math.sin(phi) * math.cos(theta),
                self.sphere_radius * math.sin(phi) * math.sin(theta),
                self.sphere_radius * math.cos(phi)
            ])

            self.particles.positions[p_index] += (
                target - self.particles.positions[p_index]
            ) * 0.1

        for idx_i, p_index in enumerate(self.disk_indices):
            r = self.disk_radii[idx_i]
            angle = self.disk_angles[idx_i] + time * (0.6 + r * 0.002) * self.rotation_speed
            h = self.disk_heights[idx_i]

            target = np.array([
                r * math.cos(angle),
                r * math.sin(angle),
                h
            ])

            self.particles.positions[p_index] += (
                target - self.particles.positions[p_index]
            ) * 0.1

        if jet_mode:

            for _ in range(10):

                pole = random.choice([-1, 1])

                spread_x = random.uniform(-5, 5)
                spread_y = random.uniform(-5, 5)

                position = np.array([
                    spread_x,
                    spread_y,
                    pole * self.sphere_radius
                ])

                direction = np.array([
                    spread_x * 0.02,
                    spread_y * 0.02,
                    pole
                ])

                direction = direction / (np.linalg.norm(direction) + 0.001)

                velocity = direction * 15

                self.jet_particles.append([position, velocity])

        new_jet_particles = []
        for position, velocity in self.jet_particles:
            position += velocity
            velocity *= 0.995

            if np.linalg.norm(position) < 1200:
                new_jet_particles.append([position, velocity])

        self.jet_particles = new_jet_particles