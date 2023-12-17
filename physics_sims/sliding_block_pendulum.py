import physics_sims
import numpy as np

class SlidingBlockPendulum(physics_sims.Sim):
    def __init__(self):
        self.t = 0

        self.x = -1.5
        self.xd = 0
        self.xdd = 0

        self.th = np.pi / 2
        self.thd = -0.8 * np.pi
        self.thdd = 0

        self.m1 = 1
        self.m2 = 1
        self.R = 3
        self.g = 9.8

    def calc_xdd(self, th, thd):
        return np.sin(th) * (self.R * thd**2 + self.g * np.cos(th)) / ((self.m1 + self.m2) / self.m2 - np.cos(th)**2)

    def calc_thdd(self, th, thd):
        return (self.m2 * thd**2 * np.cos(th) + self.g * (self.m1 + self.m2) / self.R) * np.sin(th) / (self.m2 * np.cos(th)**2 - self.m1 - self.m2)
    
    def update(self, sim_runner, dt):
        self.t, (self.x, self.th), (self.xd, self.thd) = physics_sims.integrators.runge_kutta_4th_order(
            dt,
            self.t,
            np.array([self.x, self.th]),
            np.array([self.xd, self.thd]),
            lambda _, q, qd: np.array([
                self.calc_xdd(q[1], qd[1]),
                self.calc_thdd(q[1], qd[1]),
            ])
        )

    def state(self):
        kinetic_pendulum = 0.5 * self.m2 * (
            self.R**2 * self.thd**2
            + 2 * self.R * self.xd * self.thd * np.cos(self.th)
            + self.xd**2)
        kinetic_block = 0.5 * self.m1 * self.xd**2
        potential = -self.m2 * self.g * self.R * np.cos(self.th)
        energy = potential + kinetic_block + kinetic_pendulum

        return [self.t, self.x, self.th, energy]

    def draw(self, sim_runner):
        x_block = self.x
        y_block = 0
        x_pendulum = x_block + self.R * np.sin(self.th)
        y_pendulum = y_block - self.R * np.cos(self.th)

        sim_runner.draw_dot([x_block, y_block])
        sim_runner.draw_dot([x_pendulum, y_pendulum])
        print(self.state())

trajectory = physics_sims.SimRunner().run(
    SlidingBlockPendulum(),
    time_delta=0.01,
    time_scale=1)


