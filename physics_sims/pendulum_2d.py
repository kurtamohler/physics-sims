import numpy as np
from physics_sims import SimRunner, integrators

def calc_theta_ddot(theta, g, R):
    return - (g / R) * np.sin(theta)

def calc_xy(theta, R):
        x = -R * np.sin(theta)
        y = -R * np.cos(theta)
        return np.array([x, y])

class Pendulum2DSim:
    def __init__(self, theta=3, theta_dot=0, m=10, g=9.8, R=3, *, dtype=np.float32):
        self.theta = theta
        self.theta_dot = theta_dot
        self.m = m
        self.g = g
        self.R = R

        self.theta_ddot = 0
        self.iters = 0

    def update(self, sim_runner, t, dt):
        _, self.theta, self.theta_dot, self.theta_ddot = integrators.velocity_verlet(
            dt, t, self.theta, self.theta_dot, self.theta_ddot,
            lambda _, theta, __: calc_theta_ddot(theta, self.g, self.R))

    def draw(self, sim_runner, cur_time):
        sim_runner.draw_dot(calc_xy(self.theta, self.R))
        sim_runner.draw_dot(np.array([0, 0]))

        if self.iters % 1_000 == 0:
            kinetic = 0.5 * self.m * (self.R * self.theta_dot)**2
            potential = -self.m * self.g * self.R * np.cos(self.theta)
            total = kinetic + potential
            print(f'{cur_time} {kinetic} {potential} {total}')

        self.iters += 1

if __name__ == '__main__':
    SimRunner().run(sim=Pendulum2DSim())

    
