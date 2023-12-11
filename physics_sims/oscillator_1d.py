import numpy as np
from physics_sims import SimRunner, integrators, Sim

def calc_acceleration(x, k, m):
    return -(k / m) * x

class Oscillator1DSim(Sim):
    def __init__(self, t=0, x=2, v=0, m=0.25, k=4, *, dtype=np.float64):
        self.t = np.array(t, dtype=dtype)
        self.x = np.array(x, dtype=dtype)
        self.v = np.array(v, dtype=dtype)
        self.m = np.array(m, dtype=dtype)
        self.k = np.array(k, dtype=dtype)
        self.a = calc_acceleration(x, k, m)

        self.iters = 0

    def update(self, sim_runner, dt):
        self.t, self.x, self.v, self.a = integrators.velocity_verlet(
            dt, self.t, self.x, self.v, self.a,
            lambda _, x, __: calc_acceleration(x, self.k, self.m))

    def draw(self, sim_runner):
        sim_runner.draw_dot(np.array([self.x, 0]))

        if self.iters % 1_000 == 0:
            kinetic = self.calc_kinetic()
            potential = self.calc_potential()
            print(f'{self.t}: {kinetic + potential}')

        self.iters += 1
    
    def calc_kinetic(self):
        return 0.5 * self.m * self.v**2

    def calc_potential(self):
        return 0.5 * self.k * self.x**2
    
    def state(self):
        kinetic = self.calc_kinetic()
        potential = self.calc_potential()
        return [self.t, self.x, self.v, kinetic, potential, kinetic + potential]

if __name__ == '__main__':
    SimRunner().run(sim=Oscillator1DSim(t=-99))
