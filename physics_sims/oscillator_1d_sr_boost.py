import numpy as np
from physics_sims import SimRunner

c = 1

def calc_acceleration(x, t, v, k, m, v_b):
    #return -(k / m * (1 - v_boost**2)) * (x + v_boost * t) * (1 - (v/c)**2)**(1.5)
    #return -(k / m * (1 - v_boost**2)) * (x + v_boost * t) * (1 - (v/c)**2)**(1.5)
    #return -k/m * (1 - v**2)**1.5 * (v_b * t / (1-v_b**2)**0.5 + x / (1 - v_b**2)**0.5)
    return -k/m * (1 - v**2)**1.5 * (x + v_b * t) / (1-v_b**2)**0.5

class Oscillator1DSRBoostSim:
    def __init__(self, x=2, v=0, m=0.25, k=4, v_boost=0.7, *, dtype=np.float32):
        self.x = np.array(x, dtype=dtype)
        self.v = np.array(v, dtype=dtype)
        self.m = np.array(m, dtype=dtype)
        self.k = np.array(k, dtype=dtype)
        self.v_boost = np.array(v_boost, dtype=dtype)
        self.t = np.array(0, dtype=dtype)

        self.iters = 0

    def update(self, sim_runner, cur_time, dt):
        # Classic Fourth-order Runge-Kutta integration method
        # https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
        k1v = dt * calc_acceleration(self.x, self.t, self.v, self.k, self.m, self.v_boost)
        k1x = dt * self.v

        k2v = dt * calc_acceleration(
            self.x + 0.5 * k1x,
            self.t + 0.5 * dt,
            self.v + 0.5 * k1v,
            self.k, self.m, self.v_boost)
        k2x = dt * (self.v + 0.5 * k1v)

        k3v = dt * calc_acceleration(
            self.x + 0.5 * k2x,
            self.t + 0.5 * dt,
            self.v + 0.5 * k2v,
            self.k, self.m, self.v_boost)
        k3x = dt * (self.v + 0.5 * k2v)

        k4v = dt * calc_acceleration(
            self.x + k3x,
            self.t + dt,
            self.v + k3v,
            self.k, self.m, self.v_boost)
        k4x = dt * (self.v + k3v)

        self.x = self.x + (k1x + 2.0 * (k2x + k3x) + k4x) / 6.0
        self.v = self.v + (k1v + 2.0 * (k2v + k3v) + k4v) / 6.0
        self.t = cur_time

    def draw(self, sim_runner, cur_time):
        sim_runner.draw_dot(np.array([self.x, 0]))

        if self.iters % 1_000 == 0:
            kinetic = self.calc_kinetic()
            potential = self.calc_potential()
            print(f'{cur_time}: {kinetic} {potential} {kinetic + potential}')

        self.iters += 1

    def calc_kinetic(self):
        return self.m * (1 - self.v**2)**-0.5 - self.m

    def calc_potential(self):
        return 0.5 * self.k * self.x**2
    
    def state(self, cur_time):
        kinetic = self.calc_kinetic()
        potential = self.calc_potential()
        return [cur_time, self.x, self.v, kinetic, potential, kinetic + potential]

if __name__ == '__main__':
    SimRunner().run(sim=Oscillator1DSRBoostSim(), time_scale=1)
