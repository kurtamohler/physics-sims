import physics_sims
import pygame
import numpy as np

pygame.font.init()
font = pygame.font.SysFont(
    pygame.font.get_default_font(),
    20)

class InteractiveEinsteinianSim(physics_sims.Sim):
    def __init__(self):
        self.t = 0
        self.x = 0
        self.v = 0

        self.a_player = 5
        self.c = 1
        self.t_player = 0
        self.path = [[self.t, self.x]]

        self.max_v_decimal_places = 14
        self.max_v = 1 - (0.1) ** self.max_v_decimal_places
        self.max_path_len = 10_000

        # If True, time in the rest frame passes at the same rate
        # as it does on the computer.
        # If False, time in the player's frame passes at the same
        # rate as it does on the computer.
        self.match_computer_time_to_rest_frame = False

    def calc_dt_player_by_dt(self):
        return np.sqrt(1 - (self.v / self.c)**2)

    def calc_a(self):
        return self.a_player * (1 - (self.v / self.c)**2)**(3 / 2)

    def maybe_limit_v(self):
        if self.v > self.max_v:
            self.v = self.max_v
        elif self.v < -self.max_v:
            self.v = -self.max_v

    def handle_player_controls(self, sim_runner, dt):
        keys = pygame.key.get_pressed()
        sign = None
        if keys[pygame.K_RIGHT]:
            self.v = self.v + self.calc_a() * dt
        if keys[pygame.K_LEFT]:
            self.v = self.v - self.calc_a() * dt
        if keys[pygame.K_SPACE]:
            self.v = 0

        self.maybe_limit_v()

    def update(self, sim_runner, dt_computer):
        if self.match_computer_time_to_rest_frame:
            dt = dt_computer
            dt_player = dt * self.calc_dt_player_by_dt()
        else:
            dt = dt_computer / self.calc_dt_player_by_dt()
            dt_player = dt_computer

        self.handle_player_controls(sim_runner, dt)
        self.t = self.t + dt
        self.x = self.x + self.v * dt
        self.t_player = self.t_player + dt_player
        self.path.append([self.t, self.x])

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
            pygame.draw.aaline(
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
            pygame.draw.aaline(
                sim_runner._screen,
                (0, 0, 0),
                sim_runner.convert_to_draw_position([x, t_start]),
                sim_runner.convert_to_draw_position([x, t_end]),
                1
            )

        # Path player took in the past
        path = np.asarray(self.path)
        #path = path[path[:, 0] > (self.t + t_start)]
        path = path[-self.max_path_len:]
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

        # Player's line of simultaneity: t = vx/c**2
        pygame.draw.line(
            sim_runner._screen,
            (255, 0, 0),
            sim_runner.convert_to_draw_position([x_start, self.v * x_start/self.c**2]),
            sim_runner.convert_to_draw_position([x_end, self.v * x_end/self.c**2])
        )

        # Player's line of colocality: x = vt
        pygame.draw.line(
            sim_runner._screen,
            (255, 0, 0),
            sim_runner.convert_to_draw_position([self.v * t_start, t_start]),
            sim_runner.convert_to_draw_position([self.v * t_end, t_end]),
        )

        # Player
        pygame.draw.circle(
            sim_runner._screen,
            (0, 0, 0),
            sim_runner.convert_to_draw_position([0, 0]),
            5)

        # Coordinates text
        text_list = [
            f"tau = {self.t_player:.02f}",
            f"t = {self.t:.2f}",
            f"x = {self.x:.2f}",
            f"v = {self.v:.{self.max_v_decimal_places}f}",
            f"dt/dtau = {1 / self.calc_dt_player_by_dt():.3f}"
        ]
        for idx, text in enumerate(text_list):
            sim_runner._screen.blit(
                font.render(text, True, (0, 0, 0)),
                sim_runner.convert_to_draw_position([
                    0.8 * x_start,
                    0.65 * t_start - 0.3 * idx])
            )

physics_sims.SimRunner().run(InteractiveEinsteinianSim(), time_scale=1, draw_freq=60)