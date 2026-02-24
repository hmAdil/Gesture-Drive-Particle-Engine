import pygame
import math
import cv2
import numpy as np
import os
from utils import project, rotate_x
from particles import ParticleSystem
from blackhole import BlackHoleEffect
from gestures import GestureController

pygame.init()

info = pygame.display.Info()
SCREEN_WIDTH = info.current_w
SCREEN_HEIGHT = info.current_h

os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
screen = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT),
    pygame.NOFRAME
)

clock = pygame.time.Clock()

particles = ParticleSystem()
blackhole = BlackHoleEffect(particles)
gesture = GestureController()

running = True
time = 0
previous_left_fist = False

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

    forming = gesture.left_fist
    jet_mode = gesture.left_fist and gesture.right_open

    if previous_left_fist and not gesture.left_fist:
        particles.burst()

    previous_left_fist = gesture.left_fist

    particles.drift()
    blackhole.update(forming, jet_mode, time)

    tilt = math.radians(-65)

    for pos in particles.positions:
        point = rotate_x(pos, tilt)
        px, py = project(point)

        if 0 <= px < SCREEN_WIDTH and 0 <= py < SCREEN_HEIGHT:
            screen.set_at((px, py), (255, 255, 255))

    for position, velocity in blackhole.jet_particles:
        point = rotate_x(position, tilt)
        px, py = project(point)

        if 0 <= px < SCREEN_WIDTH and 0 <= py < SCREEN_HEIGHT:
            screen.set_at((px, py), (150, 200, 255))

    pygame.display.flip()

    if gesture.debug_frame is not None:
        cv2.imshow("Gesture Debug", gesture.debug_frame)
        cv2.waitKey(1)

pygame.quit()
cv2.destroyAllWindows()