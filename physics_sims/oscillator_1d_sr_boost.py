import numpy as np
from physics_sims import SimRunner, integrators

c = 1

def calc_acceleration(t, x, v, k, m, v_b):
    return -k/m * (1 - v**2)**1.5 * (x + v_b * t) / (1-v_b**2)**0.5

class Oscillator1DSRBoostSim:
    def __init__(self, x=2, v=0, m=0.25, k=4, v_boost=0.7, *, dtype=np.float32):
        self.x = np.array(x, dtype=dtype)
        self.v = np.array(v, dtype=dtype)
        self.m = np.array(m, dtype=dtype)
        self.k = np.array(k, dtype=dtype)
        self.v_boost = np.array(v_boost, dtype=dtype)
        self.iters = 0

    def update(self, sim_runner, t, dt):
        _, self.x, self.v = integrators.runge_kutta_4th_order(
            dt,
            t,
            self.x,
            self.v,
            lambda t, x, v: calc_acceleration(t, x, v, self.k, self.m, self.v_boost))

    def draw(self, sim_runner, t):
        sim_runner.draw_dot(np.array([self.x, 0]))

        if self.iters % 1_000 == 0:
            kinetic = self.calc_kinetic()
            potential = self.calc_potential()
            print(f'{t}: {kinetic} {potential} {kinetic + potential}')

        self.iters += 1

    def calc_kinetic(self):
        return self.m * (1 - self.v**2)**-0.5 - self.m

    def calc_potential(self):
        return 0.5 * self.k * self.x**2
    
    def state(self, t):
        kinetic = self.calc_kinetic()
        potential = self.calc_potential()
        return [t, self.x, self.v, kinetic, potential, kinetic + potential]

if __name__ == '__main__':
    SimRunner().run(sim=Oscillator1DSRBoostSim(), time_scale=1)
