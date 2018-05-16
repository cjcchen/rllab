import numpy as np


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self._theta = theta
        self._mu = mu
        self._sigma = sigma
        self._dt = dt
        self._x0 = x0
        self.reset()

    def gen(self):
        x = self._x_prev + self._theta * (
            self._mu - self._x_prev) * self._dt + self._sigma * np.sqrt(
                self._dt) * np.random.normal(size=self._mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self._x_prev = self._x0 if self._x0 is not None else np.zeros_like(
            self._mu)
