import numpy as np
from physics_sims import SimRunner

def calc_acceleration(x, k, m):
    return -(k / m)

class ConstantForce1DSim:
    def __init__(self, x=0, v=0, m=0.25, k=-4, *, dtype=np.float32):
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

    def calc_kinetic(self):
        return 0.5 * self.m * self.v**2

    def calc_potential(self):
        return self.k * self.x

    def draw(self, sim_runner, cur_time):
        sim_runner.draw_dot(np.array([self.x, 0]))

        if self.iters % 1_000 == 0:
            kinetic = self.calc_kinetic()
            potential = self.calc_potential()
            print(f'{cur_time}: {kinetic + potential}')

        self.iters += 1

    def state(self, cur_time):
        kinetic = self.calc_kinetic()
        potential = self.calc_potential()
        return [cur_time, self.x, self.v, kinetic, potential, kinetic+potential]

if __name__ == '__main__':
    SimRunner().run(sim=ConstantForce1DSim())
