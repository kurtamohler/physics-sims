import physics_sims
import pygame
import numpy as np

pygame.font.init()
font = pygame.font.SysFont(
    pygame.font.get_default_font(),
    20)

def lorentz_matrix(v, c):
    lorentz_factor = (1 - (v/c)**2)**(-0.5)
    return np.array([
        [1, -v/c**2],
        [-v, 1]
    ]) * lorentz_factor

class InteractiveEinsteinianSim(physics_sims.Sim):
    def __init__(self):
        self.t = 0
        self.x = 0
        self.v = 0
        self.prev_v = self.v

        self.objects_X = np.array([
            [0., 0.],
            [0., 2.],
            [0., 0.],
            [0., 0.1],
            [0, 0],
            [0, 0],
        ])
        self.objects_v = np.array([
            0,
            0,
            0.99,
            0.99,
            0.8,
            0,
        ])
        self.objects_a_prime = [
            0,
            0,
            0,
            0,
            0,
            0.1,
        ]

        self.a_player = 1
        self.c = float('inf')
        self.c = 1
        self.t_player = 0

        self.max_v_decimal_places = 10
        self.max_v = self.c * (1 - (0.1) ** self.max_v_decimal_places)
        self.max_path_len = 10_000
        self.path = [[self.t, self.x]] * self.max_path_len

        self.dt_direction = 1
        self.prev_dt_direction = self.dt_direction

        # If True, time in the rest frame passes at the same rate
        # as it does on the computer.
        # If False, time in the player's frame passes at the same
        # rate as it does on the computer.
        self.match_computer_time_to_rest_frame = False

        # If True, draw the screen from the perspective of the player's
        # coordinate system.
        # If False, draw the screen from the perspective of the rest frame.
        self.draw_in_player_frame = False
        self.draw_in_player_frame = True

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
            dist = (t_diff ** 2 - (x_diff / self.c)**2)**0.5

            if dist > 0.1 or self.prev_dt_direction != self.dt_direction:
                # Prune path if it gets too long
                if len(self.path) >= self.max_path_len:
                    self.path = self.path[::2]
                self.path.append([self.t, self.x])

        self.prev_v = self.v
        self.prev_dt_direction = self.dt_direction

        # Simulate objects
        for idx, object_X in enumerate(self.objects_X):
            v = self.objects_v[idx]
            a_prime = self.objects_a_prime[idx]
            a = a_prime * (1 - (v / self.c)**2)**(3 / 2)
            self.objects_v[idx] += a * dt


        self.objects_X[:, 0] += float(dt)
        self.objects_X[:, 1] += self.objects_v * dt

    def draw(self, sim_runner):
        coord_range_x, coord_range_t = sim_runner.get_screen_coord_range()

        x_start = int(-(coord_range_x // 2))
        x_end = int(coord_range_x - coord_range_x // 2)

        t_start = int(-(coord_range_t // 2))
        t_end = int(coord_range_t - coord_range_t // 2)

        if self.draw_in_player_frame:
            B = lorentz_matrix(self.v, self.c)
        else:
            B = np.eye(2)

        player_event = np.matmul(
            B,
            np.array([self.t, self.x]).T).T


        # Time gridlines
        t_offset = self.t % 1
        for t_value in range(t_start + 1, t_end + 1):
            t = t_value - t_offset
            events = np.array([
                [t, x_start],
                [t, x_end]
            ])
            events = np.matmul(B, events.T).T
            events_draw = sim_runner.convert_to_draw_position(events[:, [1, 0]])
            pygame.draw.aaline(
                sim_runner._screen,
                (0, 0, 0),
                events_draw[0],
                events_draw[1],
                1
            )
            t_text = t_value + int(self.t // 1)
            sim_runner._screen.blit(
                font.render(f"{t_text:.0f}", True, (0, 0, 0)),
                (events_draw[0] + events_draw[1]) / 2)

        # Position gridlines
        x_offset = self.x % 1
        for x_value in range(x_start + 1, x_end + 1):
            x = x_value - x_offset
            events = np.array([
                [t_start, x],
                [t_end, x]
            ])
            events = np.matmul(B, events.T).T
            events_draw = sim_runner.convert_to_draw_position(events[:, [1, 0]])
            pygame.draw.aaline(
                sim_runner._screen,
                (0, 0, 0),
                events_draw[0],
                events_draw[1],
                1
            )
            x_text = x_value + int(self.x // 1)
            sim_runner._screen.blit(
                font.render(f"{x_text:.0f}", True, (0, 0, 0)),
                (events_draw[0] + events_draw[1]) / 2)

        # Player's line of simultaneity: t = vx/c**2
        if self.draw_in_player_frame:
            v_player = 0
        else:
            v_player = self.v

        pygame.draw.line(
            sim_runner._screen,
            (255, 0, 0),
            sim_runner.convert_to_draw_position([x_start, v_player * x_start/self.c**2]),
            sim_runner.convert_to_draw_position([x_end, v_player * x_end/self.c**2])
        )

        # Player's line of colocality: x = vt
        pygame.draw.line(
            sim_runner._screen,
            (255, 0, 0),
            sim_runner.convert_to_draw_position([v_player * t_start, t_start]),
            sim_runner.convert_to_draw_position([v_player * t_end, t_end]),
        )

        # Path player took in the past
        if len(self.path) >= 2:
            path = np.matmul(
                B,
                np.asarray(self.path).T).T
            path = player_event - path
            draw_path = sim_runner.convert_to_draw_position(path)
            pygame.draw.aalines(
                sim_runner._screen,
                (100, 100, 255),
                False,
                draw_path[:, [1, 0]],
                1
            )

        # Objects
        objects_X_prime = np.matmul(
            B,
            (self.objects_X - np.array([self.t, self.x])).T,
        ).T
        for object_X in objects_X_prime:
            pygame.draw.circle(
                sim_runner._screen,
                (150, 90, 40),
                sim_runner.convert_to_draw_position([object_X[1], object_X[0]]),
                5
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
