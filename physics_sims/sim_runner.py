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
            state = sim.state()
            if state is None:
                cur_time = run_time
            else:
                states.append(sim.state())

        return np.array(states)

    def run(self, sim, *, time_delta=0.001, time_scale=1):
        import pygame
        assert isinstance(sim, Sim)
        pygame.init()
        self._screen = pygame.display.set_mode(self._screen_size)

        t = time.time()
        t_sim = t
        t_last_graphics_update = -float('inf')

        t_graphics_update_period = 1 / 120

        running = True
        while running:
            if t - t_last_graphics_update >= t_graphics_update_period:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                self._screen.fill((255, 255, 255))

                sim.draw(self)

                pygame.display.flip()
                t_last_graphics_update = t
    
            while t_sim < t:
                sim.update(self, time_delta)
                t_sim += time_delta / time_scale

            t = time.time()

        pygame.quit()

    # TODO: It would be much better to separate drawing APIs like this into a
    # SimGraphics object that the SimRunner owns. `sim.update` would be given
    # that object rather than the SimRunner. That way SimRunner is not
    # responsible for implementing graphics, and also the Sim doesn't get access
    # to the SimRunner.
    def draw_dot(self, position):
        import pygame
        position = np.asarray(position)

        draw_position = self.convert_to_draw_position(position)

        pygame.draw.circle(
            self._screen,
            (0, 0, 0),
            draw_position,
            5)

    def convert_to_draw_position(self, position):
        return self._screen_size // 2 + (position - self._screen_coord_center) * (self._screen_size / self._screen_coord_scale)

    def get_screen_coord_range(self):
        return np.abs(self._screen_coord_scale)
