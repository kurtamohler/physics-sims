import numpy as np
from physics_sims import SimRunner, integrators, Sim

def calc_p_dot(x, k):
    return -k * x

def calc_x_dot(p, m):
    return p / m

class Oscillator1DPhaseSim(Sim):
    def __init__(self, t0=0, x0=2, v0=0, m=0.25, k=4, *, dtype=np.float32):
        self.x = np.array(x0, dtype=dtype)
        self.p = np.array(m * v0, dtype=dtype)
        self.m = np.array(m, dtype=dtype)
        self.k = np.array(k, dtype=dtype)
        self.t = t0

        self.iters = 0

    def update(self, sim_runner, dt):
        self.t, self.x, self.p = integrators.verlet_symplectic(
            dt, self.t, self.x, self.p,
            lambda p: calc_x_dot(p, self.m),
            lambda q: calc_p_dot(q, self.k))

    def draw(self, sim_runner):
        sim_runner.draw_dot(np.array([self.x, self.p]))

        if self.iters % 1_000 == 0:
            kinetic = 0.5 * self.p ** 2 / self.m
            potential = 0.5 * self.k * self.x**2
            print(f'{self.t}: {kinetic + potential}')

        self.iters += 1

if __name__ == '__main__':
    SimRunner().run(sim=Oscillator1DPhaseSim())
