import numpy as np
from physics_sims import Sim, SimRunner, integrators

def calc_p_dot(theta, m, g, R):
    return -m * g * R * np.sin(theta)

def calc_theta_dot(p, m, R):
    return p / (m * R**2)

def calc_xy(theta, R):
        x = -R * np.sin(theta)
        y = -R * np.cos(theta)
        return np.array([x, y])

class Pendulum2DPhaseSim(Sim):
    def __init__(self, t=0, theta0=3, theta_dot0=0, m=10, g=9.8, R=3, *, dtype=np.float32):
        self.t = t
        self.theta = theta0
        self.p = m * (R**2) * theta_dot0
        self.m = m
        self.g = g
        self.R = R

        self.iters = 0

    def update(self, sim_runner, dt):
        self.t, self.theta, self.p = integrators.verlet_symplectic(
            dt, self.t, self.theta, self.p,
            lambda p: calc_theta_dot(p, self.m, self.R),
            lambda theta: calc_p_dot(theta, self.m, self.g, self.R))

    def draw(self, sim_runner):
        sim_runner.draw_dot(calc_xy(self.theta, self.R))
        sim_runner.draw_dot(np.array([0, 0]))

        if self.iters % 1_000 == 0:
            kinetic = self.p**2 / (2 * self.m * self.R**2)
            potential = -self.m * self.g * self.R * np.cos(self.theta)
            total = kinetic + potential
            print(f'{self.t} {kinetic} {potential} {total}')

        self.iters += 1

if __name__ == '__main__':
    SimRunner().run(sim=Pendulum2DPhaseSim())

    
