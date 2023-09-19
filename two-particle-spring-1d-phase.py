import numpy as np
from sim_runner import SimRunner

def calc_x_dot(p, m):
    return p / m

def calc_p_dot(x, k, R):
    return k * (x[0] - x[1] + R) * np.array([[-1], [1]])

class TwoParticleSpring1DSim:
    def __init__(self, *, dtype=np.float32):
        self.x = np.array([[-5], [-2]], dtype=dtype)
        v = np.array([[0.001], [0]], dtype=dtype)
        self.m = np.array([[0.5], [0.5]], dtype=dtype)
        self.p = self.m * v
        self.R = np.array(1.5, dtype=dtype)
        self.k = np.array(20, dtype=dtype)

        self.iters = 0

    def update(self, sim_runner, cur_time, time_delta):
        x_halfway = self.x + 0.5 * calc_x_dot(self.p, self.m) * time_delta
        p_next = self.p + calc_p_dot(x_halfway, self.k, self.R) * time_delta
        x_next = x_halfway + 0.5 * calc_x_dot(p_next, self.m) * time_delta

        self.x = x_next
        self.p = p_next

        kinetic = (self.p**2 / (2 * self.m)).sum()
        potential = 0.5 * self.k * ((self.x[1] - self.x[0] - self.R)**2).sum()
        total = kinetic + potential

        if self.iters % 1_000 == 0:
            print(f'{cur_time} {kinetic} {potential} {total}')

        self.iters += 1

    def draw(self, sim_runner):
        sim_runner.draw_dot(np.array([self.x[0][0], 0]))
        sim_runner.draw_dot(np.array([self.x[1][0], 0]))
        sim_runner.draw_dot([self.x[0][0], self.p[0][0]])
        sim_runner.draw_dot([self.x[1][0], self.p[1][0]])

SimRunner().run(sim=TwoParticleSpring1DSim())
