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

        self._time_scale = 1 

    def run(self, sim):
        pygame.init()
        self._screen = pygame.display.set_mode(self._screen_size)

        self._start_timestamp = time.time()
        self._prev_time = 0 

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self._screen.fill((255, 255, 255))
    
            timestamp = time.time()

            cur_time = (timestamp - self._start_timestamp) * self._time_scale
            time_delta = cur_time - self._prev_time

            # TODO: I need to separate the update and draw calls. Draw only
            # needs to be called like 24 to 60 Hz. Also, I should make
            # `time_delta` consistent, because its variability is the cause of
            # occasionally huge floating point errors. If I pause the process
            # for a few seconds and bring it up again, the energy changes
            # wildly. I need to make it possible to pause the process and then
            # when it resumes, have it iterate through all the missed timesteps
            # correctly. Simulations should be repeatable. It should also be
            # possible to run a simulation in headless mode at maximum
            # processing speed.

            sim.update(self, cur_time, time_delta)
            sim.draw(self)

            pygame.display.flip()

            self._prev_time = cur_time

        pygame.quit()

    def draw_dot(self, position):
        position = np.asarray(position)

        position_pixels = self._screen_size // 2 + (position - self._screen_coord_center) * (self._screen_size / self._screen_coord_scale)

        pygame.draw.circle(
            self._screen,
            (0, 0, 0),
            position_pixels,
            5)
