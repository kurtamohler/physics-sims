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
        self.prev_v = self.v

        self.a_player = 5
        self.c = 1
        self.t_player = 0

        self.max_v_decimal_places = 12
        self.max_v = 1 - (0.1) ** self.max_v_decimal_places
        self.max_path_len = 10_000
        self.path = [[self.t, self.x]] * self.max_path_len

        self.dt_direction = 1
        self.prev_dt_direction = self.dt_direction

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

    def handle_player_controls(self, dt):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_RIGHT]:
            self.v = self.v + self.calc_a() * dt
        if keys[pygame.K_LEFT]:
            self.v = self.v - self.calc_a() * dt

        if keys[pygame.K_UP]:
            self.dt_direction = 1
        if keys[pygame.K_DOWN]:
            self.dt_direction = -1

        if keys[pygame.K_SPACE]:
            self.v = 0

        self.maybe_limit_v()

    def update(self, sim_runner, dt_computer):
        if self.match_computer_time_to_rest_frame:
            dt = self.dt_direction * dt_computer
            dt_player = dt * self.calc_dt_player_by_dt()
        else:
            dt = self.dt_direction * dt_computer / self.calc_dt_player_by_dt()
            dt_player = dt_computer

        self.handle_player_controls(dt)
        self.t = self.t + dt
        self.x = self.x + self.v * dt
        self.t_player = self.t_player + dt_player

        if self.prev_v == self.v and self.prev_dt_direction == self.dt_direction:
            self.path[-1] = [self.t, self.x]
        else:
            # Only add to the path if the distance between events is noticeable
            # or if dt direction changes
            x_diff = self.x - self.path[-2][1]
            t_diff = self.t - self.path[-2][0]
            dist = (t_diff ** 2 - x_diff**2)**0.5

            if dist > 0.1 or self.prev_dt_direction != self.dt_direction:
                # Prune path if it gets too long
                if len(self.path) >= self.max_path_len:
                    self.path = self.path[::2]
                self.path.append([self.t, self.x])

        self.prev_v = self.v
        self.prev_dt_direction = self.dt_direction

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
        if len(self.path) >= 2:
            path = np.array([self.t, self.x]) - np.asarray(self.path)
            draw_path = sim_runner.convert_to_draw_position(path)
            pygame.draw.aalines(
                sim_runner._screen,
                (100, 100, 255),
                False,
                draw_path[:, [1, 0]],
                1
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
            f"dt/dtau = {1 / self.calc_dt_player_by_dt():.3f}",
            f"path len: {len(self.path)}"
        ]
        for idx, text in enumerate(text_list):
            sim_runner._screen.blit(
                font.render(text, True, (0, 0, 0)),
                sim_runner.convert_to_draw_position([
                    0.8 * x_start,
                    0.65 * t_start - 0.3 * idx])
            )

physics_sims.SimRunner().run(InteractiveEinsteinianSim(), time_scale=1, draw_freq=60)