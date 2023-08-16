import numpy as np
from sim_runner import SimRunner

class TwoParticleSpring1DSim:

    def calc_a1(self, x1, x2):
        return -(self.k / self.m1) * (x1 - x2 + self.R)

    def calc_a2(self, x1, x2):
        return (self.k / self.m2) * (x1 - x2 + self.R)

    def __init__(self, *, dtype=np.float32):
        self.x1 = np.array(-5, dtype=dtype)
        self.x2 = np.array(-3, dtype=dtype)
        self.v1 = np.array(0.2, dtype=dtype)
        self.v2 = np.array(0, dtype=dtype)
        self.m1 = 1
        self.m2 = 1
        self.R = 1.5
        self.k = 200

        self.a1 = self.calc_a1(self.x1, self.x2)
        self.a2 = self.calc_a2(self.x1, self.x2)

        self.iters = 0

    def update(self, sim_runner, cur_time, time_delta):
        x1_next = self.x1 + self.v1 * time_delta + self.a1 * (time_delta**2) * 0.5
        x2_next = self.x2 + self.v2 * time_delta + self.a2 * (time_delta**2) * 0.5

        a1_next = self.calc_a1(x1_next, x2_next)
        a2_next = self.calc_a2(x1_next, x2_next)

        v1_next = self.v1 + (self.a1 + a1_next) * time_delta * 0.5
        v2_next = self.v2 + (self.a2 + a2_next) * time_delta * 0.5

        self.x1 = x1_next
        self.v1 = v1_next
        self.a1 = a1_next

        self.x2 = x2_next
        self.v2 = v2_next
        self.a2 = a2_next

        kinetic = 0.5 * self.m1 * self.v1**2 + 0.5 * self.m2 * self.v2**2
        potential = 0.5 * self.k * (self.x2 - self.x1 - self.R)**2
        total = kinetic + potential

        if self.iters % 1_000 == 0:
            print(f'{cur_time} {kinetic} {potential} {total}')

        self.iters += 1

    def draw(self, sim_runner):
        sim_runner.draw_dot(np.array([self.x1, 0]))
        sim_runner.draw_dot(np.array([self.x2, 0]))

SimRunner().run(sim=TwoParticleSpring1DSim())
