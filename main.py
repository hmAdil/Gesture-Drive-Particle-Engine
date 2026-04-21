import pygame
import math
import cv2
import numpy as np
import os
from utils import rotate_x_batch, project_batch
from particles import ParticleSystem
from blackhole import BlackHoleEffect
from vortex import VortexEffect
from shockwave import ShockwaveEffect
from expansion import ExpansionEffect
from gestures import GestureController

pygame.init()

info = pygame.display.Info()
SCREEN_WIDTH  = info.current_w
SCREEN_HEIGHT = info.current_h

os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
screen = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT),
    pygame.NOFRAME
)

clock = pygame.time.Clock()

particles  = ParticleSystem()
blackhole  = BlackHoleEffect(particles)
vortex     = VortexEffect(particles)
shockwave  = ShockwaveEffect(particles)
expansion  = ExpansionEffect(particles)
gesture    = GestureController()

running = True
time    = 0

# ── Edge-detection state ──────────────────────────────────────────────────────
prev_left_fist  = False   # burst on release
prev_left_open  = False   # shockwave fires on rising edge

TILT = math.radians(-65)

while running:
    clock.tick(60)
    screen.fill((6, 6, 18))
    time += 0.01

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # ── Read gesture states ───────────────────────────────────────────────────
    forming    = gesture.left_fist
    jet_mode   = gesture.left_fist and gesture.right_open
    vortex_on  = gesture.right_fist
    shock_edge = gesture.left_open and not prev_left_open   # rising edge only
    expand_on  = gesture.both_open

    # ── Edge-triggered effects ────────────────────────────────────────────────
    if prev_left_fist and not gesture.left_fist:
        particles.burst()

    if shock_edge:
        shockwave.trigger()

    prev_left_fist = gesture.left_fist
    prev_left_open = gesture.left_open

    # ── Update physics and effects ────────────────────────────────────────────
    particles.drift()
    blackhole.update(forming, jet_mode, time)
    vortex.update(vortex_on, time)
    expansion.update(expand_on)

    # ── Render: main particles ────────────────────────────────────────────────
    rotated = rotate_x_batch(particles.positions, TILT)
    px, py  = project_batch(rotated)

    mask = (px >= 0) & (px < SCREEN_WIDTH) & (py >= 0) & (py < SCREEN_HEIGHT)

    # Colour hint: tint particles slightly when effects are active
    if vortex_on:
        particle_color = (180, 255, 220)   # cyan-green for vortex
    elif expand_on:
        particle_color = (255, 220, 140)   # warm amber for expansion
    elif forming:
        particle_color = (255, 255, 255)   # white for black hole
    else:
        particle_color = (255, 255, 255)

    for x, y in zip(px[mask], py[mask]):
        screen.set_at((x, y), particle_color)

    # ── Render: jet particles (black hole) ────────────────────────────────────
    if blackhole.jet_particles:
        jet_pos = np.array([p for p, _ in blackhole.jet_particles])
        jet_rot = rotate_x_batch(jet_pos, TILT)
        jpx, jpy = project_batch(jet_rot)

        jmask = (jpx >= 0) & (jpx < SCREEN_WIDTH) & (jpy >= 0) & (jpy < SCREEN_HEIGHT)
        for x, y in zip(jpx[jmask], jpy[jmask]):
            screen.set_at((x, y), (150, 200, 255))

    pygame.display.flip()

    if gesture.debug_frame is not None:
        cv2.imshow("Gesture Debug", gesture.debug_frame)
        cv2.waitKey(1)

pygame.quit()
cv2.destroyAllWindows()
