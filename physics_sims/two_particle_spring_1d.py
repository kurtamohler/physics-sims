import numpy as np
from physics_sims import Sim, SimRunner, integrators

class TwoParticleSpring1DSim(Sim):
    def calc_a(self, x):
        return self.k * (x[0] - x[1] + self.R) * np.array([[-1], [1]]) / self.m

    def __init__(self, *, dtype=np.float32):
        self.t = 0
        self.x = np.array([[-5], [-2]], dtype=dtype)
        self.v = np.array([[0.1], [0]], dtype=dtype)
        self.m = np.array([[0.5], [0.5]], dtype=dtype)
        self.R = np.array(1.5, dtype=dtype)
        self.k = np.array(20, dtype=dtype)

        self.a = self.calc_a(self.x)

        self.iters = 0

    def update(self, sim_runner, dt):
        self.t, self.x, self.v, self.a = integrators.velocity_verlet(
            dt, self.t, self.x, self.v, self.a,
            lambda _, x, __: self.calc_a(x))

    def draw(self, sim_runner):
        sim_runner.draw_dot(np.array([self.x[0][0], 0]))
        sim_runner.draw_dot(np.array([self.x[1][0], 0]))

        kinetic = (0.5 * self.m * self.v**2).sum()
        potential = 0.5 * self.k * ((self.x[1] - self.x[0] - self.R)**2).sum()
        total = kinetic + potential

        if self.iters % 1_000 == 0:
            print(f'{self.t} {kinetic} {potential} {total}')

        self.iters += 1

if __name__ == '__main__':
    SimRunner().run(sim=TwoParticleSpring1DSim())
