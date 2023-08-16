import numpy as np
from sim_runner import SimRunner

class TwoParticleSpring1DSim:
    def calc_a(self, x):
        return self.k * (x[0] - x[1] + self.R) * np.array([[-1], [1]]) / self.m

    def __init__(self, *, dtype=np.float32):
        self.x = np.array([[-5], [-3]], dtype=dtype)
        self.v = np.array([[0.1], [0]], dtype=dtype)
        self.m = np.array([[5], [1]], dtype=dtype)
        self.R = np.array(1.5, dtype=dtype)
        self.k = np.array(200, dtype=dtype)

        self.a = self.calc_a(self.x)

        self.iters = 0

    def update(self, sim_runner, cur_time, time_delta):
        x_next = self.x + self.v * time_delta + self.a * (time_delta**2) * 0.5
        a_next = self.calc_a(x_next)
        v_next = self.v + (self.a + a_next) * time_delta * 0.5

        self.x = x_next
        self.v = v_next
        self.a = a_next

        kinetic = (0.5 * self.m * self.v**2).sum()
        potential = 0.5 * self.k * ((self.x[1] - self.x[0] - self.R)**2).sum()
        total = kinetic + potential

        if self.iters % 1_000 == 0:
            print(f'{cur_time} {kinetic} {potential} {total}')

        self.iters += 1

    def draw(self, sim_runner):
        sim_runner.draw_dot(np.array([self.x[0][0], 0]))
        sim_runner.draw_dot(np.array([self.x[1][0], 0]))

SimRunner().run(sim=TwoParticleSpring1DSim())
