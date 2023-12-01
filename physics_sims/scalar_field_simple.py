import numpy as np
from physics_sims import SimRunner

c = 1

def calc_acceleration(x, v, k, m):
    return -(k * (1 - (v/c)**2)) / (m + (k * x / c**2))

class ScalarFieldSimple1DSim:
    def __init__(self, x=0, v=0, m=0.25, k=-0.5, *, dtype=np.float32):
        self.x = np.array(x, dtype=dtype)
        self.v = np.array(v, dtype=dtype)
        self.m = np.array(m, dtype=dtype)
        self.k = np.array(k, dtype=dtype)

        self.iters = 0

    def update(self, sim_runner, cur_time, dt):
        # Classic Fourth-order Runge-Kutta integration method
        # https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
        k1v = dt * calc_acceleration(self.x, self.v, self.k, self.m)
        k1x = dt * self.v

        k2v = dt * calc_acceleration(
            self.x + 0.5 * k1x,
            self.v + 0.5 * k1v,
            self.k, self.m)
        k2x = dt * (self.v + 0.5 * k1v)

        k3v = dt * calc_acceleration(
            self.x + 0.5 * k2x,
            self.v + 0.5 * k2v,
            self.k, self.m)
        k3x = dt * (self.v + 0.5 * k2v)

        k4v = dt * calc_acceleration(
            self.x + k3x,
            self.v + k3v,
            self.k, self.m)
        k4x = dt * (self.v + k3v)

        self.x = self.x + (k1x + 2.0 * (k2x + k3x) + k4x) / 6.0
        self.v = self.v + (k1v + 2.0 * (k2v + k3v) + k4v) / 6.0


    def calc_kinetic(self):
        return -self.m * c**2 * ((1 - (self.v / c)**2)**0.5 - 1)

    def calc_potential(self):
        phi = self.k * self.x / c**2
        return phi * (1 - (self.v / c)**2)**0.5
        

    def draw(self, sim_runner, cur_time):
        sim_runner.draw_dot(np.array([self.x, 0]))
        if self.iters % 1_000 == 0:
            kinetic = self.calc_kinetic()
            potential = self.calc_potential()
            #print(f'{cur_time}: {kinetic} {potential} {kinetic + potential}')
            print(f'{cur_time}: {kinetic} {potential} {kinetic + potential} {self.v}')

        self.iters += 1

    def state(self, cur_time):
        kinetic = self.calc_kinetic()
        potential = self.calc_potential()
        return [cur_time, self.x, self.v, kinetic, potential, kinetic + potential]

if __name__ == '__main__':
    SimRunner().run(ScalarFieldSimple1DSim(), time_delta=0.0001)
