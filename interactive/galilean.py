import physics_sims
import pygame
import numpy as np

pygame.font.init()
font = pygame.font.SysFont(
    pygame.font.get_default_font(),
    20)

class InteractiveGalileanSim(physics_sims.Sim):
    def __init__(self):
        self.t = 0
        self.x = 0
        self.v = 0
        self.a = 10
        self.path = [[self.t, self.x]]

    def update(self, sim_runner, dt):
        self.path.append([self.t, self.x])
        self.handle_player_controls(sim_runner, dt)
        self.t = self.t + dt
        self.x = self.x + self.v * dt

    def handle_player_controls(self, sim_runner, dt):
        keys=pygame.key.get_pressed()
        # TODO: Ramp up to the maximum acceleration over a short period of time
        if keys[pygame.K_RIGHT]:
            self.v = self.v + self.a * dt
        if keys[pygame.K_LEFT]:
            self.v = self.v - self.a * dt

    def draw(self, sim_runner):
        coord_range_x, coord_range_t = sim_runner.get_screen_coord_range()

        x_start = int(-(coord_range_x // 2))
        x_end = int(coord_range_x - coord_range_x // 2)

        t_start = int(-(coord_range_t // 2))
        t_end = int(coord_range_t - coord_range_t // 2)

        # Time gridlines
        t_offset = self.t % 1
        for t in range(t_start, t_end + 1):
            t -= t_offset
            pygame.draw.line(
                sim_runner._screen,
                (0, 0, 0),
                sim_runner.convert_to_draw_position([x_start, t]),
                sim_runner.convert_to_draw_position([x_end, t]),
                1
            )

        # Position gridlines
        x_offset = self.x % 1
        for x in range(x_start, x_end + 1):
            x -= x_offset
            pygame.draw.line(
                sim_runner._screen,
                (0, 0, 0),
                sim_runner.convert_to_draw_position([x, t_start]),
                sim_runner.convert_to_draw_position([x, t_end]),
                1
            )

        # Path player took in the past
        path = np.asarray(self.path)
        path = path[path[:, 0] > (self.t + t_start)]
        # Replace the path with the truncated one
        self.path = path.tolist()
        if len(self.path) >= 2:
            path = np.array([self.t, self.x]) - path
            draw_path = sim_runner.convert_to_draw_position(path)
            pygame.draw.aalines(
                sim_runner._screen,
                (100, 100, 255),
                False,
                draw_path[:, [1, 0]],
                2
            )

        # Player's line of simultaneity
        pygame.draw.line(
            sim_runner._screen,
            (255, 0, 0),
            sim_runner.convert_to_draw_position([x_start, 0]),
            sim_runner.convert_to_draw_position([x_end, 0])
        )

        # Player
        pygame.draw.circle(
            sim_runner._screen,
            (0, 0, 0),
            sim_runner.convert_to_draw_position([0, 0]),
            5)

        # Coordinates text
        text_list = [
            f"t = {self.t:.02f}",
            f"x = {self.x:.02f}",
            f"v = {self.v:.03f}",
        ]
        for idx, text in enumerate(text_list):
            sim_runner._screen.blit(
                font.render(text, True, (0, 0, 0)),
                sim_runner.convert_to_draw_position([
                    0.8 * x_start,
                    0.7 * t_start - 0.3 * idx])
            )

physics_sims.SimRunner().run(InteractiveGalileanSim(), time_scale=1)