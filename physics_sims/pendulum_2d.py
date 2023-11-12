import numpy as np
from physics_sims import SimRunner

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

        self.theta_ddot = calc_theta_ddot(theta, g, R)
        self.iters = 0

    def update(self, sim_runner, cur_time, time_delta):
        theta_next = self.theta + self.theta_dot * time_delta + self.theta_ddot * (time_delta**2) * 0.5
        theta_ddot_next = calc_theta_ddot(theta_next, self.g, self.R)
        theta_dot_next = self.theta_dot + (self.theta_ddot + theta_ddot_next) * time_delta * 0.5

        self.theta = theta_next
        self.theta_dot = theta_dot_next
        self.theta_ddot = theta_ddot_next

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

    
