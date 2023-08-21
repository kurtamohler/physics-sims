import numpy as np
from sim_runner import SimRunner

class DoublePendulum2DSim:

    def __init__(self, *, dtype=np.float32):
        self.theta = np.array([np.pi/2, np.pi], dtype=dtype)
        self.omega = np.array([0, 0], dtype=dtype)

        self.m = np.array([1, 1], dtype=dtype)
        self.R = np.array([2, 2], dtype=dtype)
        self.g = 9.80

        self.alpha = self.calc_alpha(self.theta, self.omega)

        self.iters = 0

    # This is based on my derivation of the Euler-Lagrange equations of the
    # double pendulum.
    #
    # PERF: I did not try to simplify the equations much, so it's
    # very likely that the numerical accuracy can be improved.
    def calc_alpha(self, theta, omega):
        dL_dtheta0 = (
            -self.g * self.R[0] * (self.m.sum()) * np.sin(theta[0])
            - self.m[1] * self.R.prod() * omega.prod() * np.sin(theta[0] - theta[1]))

        dL_dtheta1 = (
            -self.g * self.R[1] * self.m[1] * np.sin(theta[1])
            + self.m[1] * self.R.prod() * omega.prod() * np.sin(theta[0] - theta[1]))

        A = self.m[1] * self.R[1]**2
        B = self.m[1] * self.R.prod()
        C = np.cos(theta[0] - theta[1])
        D = np.sin(theta[0] - theta[1])
        E = self.m.sum() * self.R[0]**2

        alpha0 = (A * dL_dtheta0 - B * C * dL_dtheta1 + D * (A * B * omega[1] - B**2 * C * omega[0]) * (omega[0] - omega[1])) / (A * E - B**2 * C**2)

        # PERF: This is a good candidate for numerical improvement, to remove
        # dependence on `alpha0`
        alpha1 = (dL_dtheta1 - B * (alpha0 * C - omega[0] * (omega[0] - omega[1]) * D)) / A

        return np.array([alpha0, alpha1], dtype=theta.dtype)

    def calc_energy(self):
        kinetic = (
            0.5 * self.m.sum() * self.R[0]**2 * self.omega[0]**2
            + 0.5 * self.m[1] * self.R[1]**2 * self.omega[1]**2
            + self.m[1] * self.R.prod() * self.omega.prod() * np.cos(self.theta[0] - self.theta[1]))

        potential = (
            -self.g * self.R[0] * self.m.sum() * np.cos(self.theta[0])
            - self.g * self.R[1] * self.m[1] * np.cos(self.theta[1]))

        return kinetic, potential

    def update(self, sim_runner, cur_time, time_delta):
        theta_next = self.theta + self.omega * time_delta + self.alpha * (time_delta**2) * 0.5
        alpha_next = self.calc_alpha(theta_next, self.omega)
        omega_next = self.omega + (self.alpha + alpha_next) * time_delta * 0.5

        self.theta = theta_next
        self.omega = omega_next
        self.alpha = alpha_next

        kinetic, potential = self.calc_energy()
        total = kinetic + potential
        print(f'{cur_time} {kinetic} {potential} {total}')

        self.iters += 1

    def draw(self, sim_runner):
        x0 = self.R[0] * np.sin(self.theta[0])
        y0 = -self.R[0] * np.cos(self.theta[0])

        x1 = x0 + self.R[1] * np.sin(self.theta[1])
        y1 = y0 - self.R[1] * np.cos(self.theta[1])

        sim_runner.draw_dot([0, 0])
        sim_runner.draw_dot([x0, y0])
        sim_runner.draw_dot([x1, y1])

SimRunner().run(sim=DoublePendulum2DSim(), time_delta=0.0001)
