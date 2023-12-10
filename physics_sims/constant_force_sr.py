import numpy as np
from physics_sims import SimRunner, integrators

c = 1

def calc_acceleration(x, v, k, m):
    return -(k / m) * (1 - (v/c)**2)**(1.5)

class ConstantForceSR1DSim:
    def __init__(self, x=0, v=0, m=0.25, k=-0.1, *, dtype=np.float32):
        self.x = np.array(x, dtype=dtype)
        self.v = np.array(v, dtype=dtype)
        self.m = np.array(m, dtype=dtype)
        self.k = np.array(k, dtype=dtype)

        self.iters = 0

    def update(self, sim_runner, t, dt):
        _, self.x, self.v = integrators.runge_kutta_4th_order(
            dt, t, self.x, self.v,
            lambda _, x, v : calc_acceleration(x, v, self.k, self.m))

    def calc_kinetic(self):
        return self.m * self.v**2 / (1 - (self.v / c)**2)**0.5 + self.m * c**2 * ((1 - (self.v / c)**2)**0.5 - 1)

    def calc_potential(self):
        return self.k * self.x

    def draw(self, sim_runner, cur_time):
        sim_runner.draw_dot(np.array([self.x, 0]))
        if self.iters % 1_000 == 0:
            kinetic = self.calc_kinetic()
            potential = self.calc_potential()
            print(f'{cur_time}: {kinetic} {potential} {kinetic + potential} {self.v}')

        self.iters += 1

    def state(self, cur_time):
        kinetic = self.calc_kinetic()
        potential = self.calc_potential()
        return [cur_time, self.x, self.v, kinetic, potential, kinetic + potential]

if __name__ == '__main__':
    SimRunner().run(ConstantForceSR1DSim())
