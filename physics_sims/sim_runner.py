import numpy as np
import time

from physics_sims import Sim

class SimRunner:
    def __init__(self):
        # The size of the screen in pixels
        self._screen_size = np.array([500, 500])

        # The scale of the simulation coordinates spanning across the screen
        self._screen_coord_scale = np.array([10, -10])

        # The simulation coordinate of the center point of the screen
        self._screen_coord_center = np.array([0, 0]) 

    def run_headless(self, sim, run_time, *, time_delta=0.001):
        assert isinstance(sim, Sim)
        cur_time = 0
        states = [sim.state()]

        while cur_time < run_time:
            cur_time += time_delta
            sim.update(self, time_delta)
            states.append(sim.state())

        return np.array(states)

    def run(self, sim, *, time_delta=0.001, time_scale=1):
        import pygame
        assert isinstance(sim, Sim)
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

            cur_time = self._cur_time

            # TODO: Limit the draw rate
            sim.draw(self)
    
            timestamp = time.time()
            target_time = (timestamp - self._start_timestamp) * time_scale

            while cur_time < target_time:
                sim.update(self, time_delta)
                cur_time += time_delta

            self._cur_time = cur_time

            pygame.display.flip()

        pygame.quit()

    # TODO: It would be much better to separate drawing APIs like this into a
    # SimGraphics object that the SimRunner owns. `sim.update` would be given
    # that object rather than the SimRunner. That way SimRunner is not
    # responsible for implementing graphics, and also the Sim doesn't get access
    # to the SimRunner.
    def draw_dot(self, position):
        import pygame
        position = np.asarray(position)

        position_pixels = self._screen_size // 2 + (position - self._screen_coord_center) * (self._screen_size / self._screen_coord_scale)

        pygame.draw.circle(
            self._screen,
            (0, 0, 0),
            position_pixels,
            5)
