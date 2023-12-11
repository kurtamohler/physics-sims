import numpy as np
from physics_sims import Sim, SimRunner, integrators

c = 1

def calc_acceleration(x, v, k, m):
    return -k * (1 - v**2) / (m + k * x)

class ScalarFieldSimple1DSim(Sim):
    def __init__(self, t=0, x=0, v=0, m=0.25, k=-0.1, *, dtype=np.float32):
        self.t = t
        self.x = np.array(x, dtype=dtype)
        self.v = np.array(v, dtype=dtype)
        self.m = np.array(m, dtype=dtype)
        self.k = np.array(k, dtype=dtype)

        self.iters = 0

    def update(self, sim_runner, dt):
        self.t, self.x, self.v = integrators.runge_kutta_4th_order(
            dt, self.t, self.x, self.v,
            lambda _, x, v: calc_acceleration(x, v, self.k, self.m))

    def calc_kinetic(self):
        return -self.m * c**2 * ((1 - (self.v / c)**2)**0.5 - 1)

    def calc_potential(self):
        phi = self.k * self.x / c**2
        return phi * (1 - (self.v / c)**2)**0.5
        
    def draw(self, sim_runner):
        sim_runner.draw_dot(np.array([self.x, 0]))
        if self.iters % 1_000 == 0:
            kinetic = self.calc_kinetic()
            potential = self.calc_potential()
            print(f'{self.t}: {kinetic} {potential} {kinetic + potential} {self.v}')

        self.iters += 1

    def state(self):
        kinetic = self.calc_kinetic()
        potential = self.calc_potential()
        return [self.t, self.x, self.v, kinetic, potential, kinetic + potential]

if __name__ == '__main__':
    SimRunner().run(ScalarFieldSimple1DSim(), time_delta=0.0001)
