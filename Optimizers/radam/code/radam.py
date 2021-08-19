# based on https://arxiv.org/pdf/1908.03265.pdf

import numpy as np


class RAdam:
    """RAdam
    Parameters:
    -----------
    learning_rate: float = 0.001
        The step length used when following the negative gradient.
    beta_1: float = 0.9
        The exponential decay rate for the 1st moment estimates.
    beta_2: float = 0.999
        The exponential decay rate for the 2nd moment estimates.    
    """
    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999) -> None:
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.p_max = 2 / (1 - self.beta_2) - 1

        self.t = 0
        self.m = None  # Decaying averages of past gradients
        self.v = None  # Decaying averages of past squared gradients

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.m is None:
            self.m = np.zeros(np.shape(grad_wrt_w))
            self.v = np.zeros(np.shape(grad_wrt_w))

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_wrt_w
        self.v = 1 / self.beta_2 * self.v + (1 - self.beta_2) * np.power(grad_wrt_w, 2)

        m_hat = self.m / (1 - self.beta_1**self.t)
        p_t = self.p_max - 2 * self.t * self.beta_2**self.t / (1 - self.beta_2**self.t)

        if p_t > 4:
            l_t = np.sqrt((1 - self.beta_2**self.t) / self.v)
            r_t = np.sqrt(((p_t - 4) * (p_t - 2) * self.p_max) / ((self.p_max - 4) * (self.p_max - 2) * p_t))
            w_update = self.learning_rate * r_t * m_hat * l_t
        else:
            w_update = self.learning_rate * m_hat

        return w - w_update
