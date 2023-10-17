import pygame
import numpy as np
import time

class SimRunner:
    def __init__(self):
        # The size of the screen in pixels
        self._screen_size = np.array([500, 500])

        # The scale of the simulation coordinates spanning across the screen
        self._screen_coord_scale = np.array([10, -10])

        # The simulation coordinate of the center point of the screen
        self._screen_coord_center = np.array([0, 0]) 

    def run_headless(self, sim, run_time, *, time_delta=0.001):
        cur_time = 0
        states = [sim.state(cur_time)]

        while cur_time < run_time:
            cur_time += time_delta
            sim.update(self, cur_time, time_delta)
            states.append(sim.state(cur_time))

        return states

    def run(self, sim, *, time_delta=0.001, time_scale=1):
        pygame.init()
        self._screen = pygame.display.set_mode(self._screen_size)

        self._start_timestamp = time.time()
        self._cur_time = 0

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self._screen.fill((255, 255, 255))
    
            timestamp = time.time()
            target_time = (timestamp - self._start_timestamp) * time_scale

            cur_time = self._cur_time

            while cur_time < target_time:
                sim.update(self, cur_time, time_delta)
                cur_time += time_delta

            self._cur_time = cur_time

            # TODO: Only draw at a fixed rate
            sim.draw(self, cur_time)

            pygame.display.flip()

        pygame.quit()

    def draw_dot(self, position):
        position = np.asarray(position)

        position_pixels = self._screen_size // 2 + (position - self._screen_coord_center) * (self._screen_size / self._screen_coord_scale)

        pygame.draw.circle(
            self._screen,
            (0, 0, 0),
            position_pixels,
            5)
