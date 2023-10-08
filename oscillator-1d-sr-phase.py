import numpy as np
from sim_runner import SimRunner

c = 1

def calc_p_dot(x, k):
    return -k * x

def calc_x_dot(p, m):
    return p * (m**2 + p**2)**(-0.5)

class Oscillator1DSim:
    def __init__(self, x=2, v=0, m=0.25, k=4, *, dtype=np.float32):
        self.x = np.array(x, dtype=dtype)
        self.p = np.array(m * v * (1 - v**2)**(-0.5), dtype=dtype)
        self.m = np.array(m, dtype=dtype)
        self.k = np.array(k, dtype=dtype)

        self.iters = 0

    def update(self, sim_runner, cur_time, dt):
        # Use Verlet symplectic integrator to limit energy loss:
        # https://en.wikipedia.org/wiki/Symplectic_integrator#A_second-order_example
        x_halfway = self.x + 0.5 * calc_x_dot(self.p, self.m) * dt
        p_next = self.p + calc_p_dot(x_halfway, self.k) * dt
        x_next = x_halfway + 0.5 * calc_x_dot(p_next, self.m) * dt

        self.x = x_next
        self.p = p_next

        if self.iters % 1_000 == 0:
            kinetic = 0.5 * self.p ** 2 / self.m
            potential = 0.5 * self.k * self.x**2

            kinetic = (self.m**2 + self.p**2)**0.5 - self.m
            potential = 0.5 * self.k * self.x**2
            print(f'{cur_time}: {kinetic + potential}')

        self.iters += 1

    def draw(self, sim_runner):
        sim_runner.draw_dot(np.array([self.x, 0]))

SimRunner().run(sim=Oscillator1DSim())
