import numpy as np
from sim_runner import SimRunner

def calc_acceleration(x, k, m):
    return -(k / m) * x

class Oscillator1DSim:
    def __init__(self, x=2, v=0, m=0.25, k=4, *, dtype=np.float32):
        self.x = np.array(x, dtype=dtype)
        self.v = np.array(v, dtype=dtype)
        self.m = np.array(m, dtype=dtype)
        self.k = np.array(k, dtype=dtype)
        self.a = calc_acceleration(x, k, m)

        self.iters = 0

    def update(self, sim_runner, cur_time, time_delta):
        # Use velocity Verlet integration to limit energy loss:
        # https://en.wikipedia.org/wiki/Verlet_integration
        x_next = self.x + self.v * time_delta + self.a * (time_delta**2) * 0.5
        a_next = calc_acceleration(x_next, self.k, self.m)
        v_next = self.v + (self.a + a_next) * time_delta * 0.5

        self.x = x_next
        self.v = v_next
        self.a = a_next

        if self.iters % 1_000 == 0:
            kinetic = 0.5 * self.m * self.v**2
            potential = 0.5 * self.k * self.x**2
            print(f'{cur_time}: {kinetic + potential}')

        self.iters += 1

    def draw(self, sim_runner):
        sim_runner.draw_dot(np.array([self.x, 0]))

SimRunner().run(sim=Oscillator1DSim())
