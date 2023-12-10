import numpy as np
from physics_sims import SimRunner, integrators

def calc_x_dot(p, m):
    return p / m

def calc_p_dot(x, k, R):
    return k * (x[0] - x[1] + R) * np.array([[-1], [1]])

class TwoParticleSpring1DPhaseSim:
    def __init__(self, *, dtype=np.float32):
        self.x = np.array([[-5], [-2]], dtype=dtype)
        v = np.array([[0.1], [0]], dtype=dtype)
        self.m = np.array([[0.5], [0.5]], dtype=dtype)
        self.p = self.m * v
        self.R = np.array(1.5, dtype=dtype)
        self.k = np.array(20, dtype=dtype)

        self.iters = 0

    def update(self, sim_runner, t, dt):
        _, self.x, self.p = integrators.verlet_symplectic(
            dt, t, self.x, self.p,
            lambda p: calc_x_dot(p, self.m),
            lambda x: calc_p_dot(x, self.k, self.R))

    def draw(self, sim_runner, t):
        sim_runner.draw_dot(np.array([self.x[0][0], 0]))
        sim_runner.draw_dot(np.array([self.x[1][0], 0]))
        sim_runner.draw_dot([self.x[0][0], self.p[0][0]])
        sim_runner.draw_dot([self.x[1][0], self.p[1][0]])

        if self.iters % 1_000 == 0:
            kinetic = (self.p**2 / (2 * self.m)).sum()
            potential = 0.5 * self.k * ((self.x[1] - self.x[0] - self.R)**2).sum()
            total = kinetic + potential
            print(f'{t} {kinetic} {potential} {total}')

        self.iters += 1

if __name__ == '__main__':
    SimRunner().run(sim=TwoParticleSpring1DPhaseSim())
