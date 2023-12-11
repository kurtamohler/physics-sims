import numpy as np
from physics_sims import SimRunner, integrators, Sim

c = 1

def calc_p_dot(x, k):
    return -k * x

def calc_x_dot(p, m):
    return p * (m**2 + p**2)**(-0.5)

class Oscillator1DSRPhaseSim(Sim):
    def __init__(self, t=0, x=2, v=0, m=0.25, k=4, *, dtype=np.float32):
        self.x = np.array(x, dtype=dtype)
        self.p = np.array(m * v * (1 - v**2)**(-0.5), dtype=dtype)
        self.m = np.array(m, dtype=dtype)
        self.k = np.array(k, dtype=dtype)
        self.t = t

        self.iters = 0

    def update(self, sim_runner, dt):
        self.t, self.x, self.p = integrators.verlet_symplectic(
            dt, self.t, self.x, self.p,
            lambda p: calc_x_dot(p, self.m),
            lambda q: calc_p_dot(q, self.k))

    def draw(self, sim_runner):
        sim_runner.draw_dot(np.array([self.x, self.p/5]))

        if self.iters % 1_000 == 0:
            kinetic = 0.5 * self.p ** 2 / self.m
            potential = 0.5 * self.k * self.x**2

            kinetic = (self.m**2 + self.p**2)**0.5 - self.m
            potential = 0.5 * self.k * self.x**2
            print(f'{self.t}: {kinetic + potential}')

        self.iters += 1

if __name__ == '__main__':
    SimRunner().run(sim=Oscillator1DSRPhaseSim())
