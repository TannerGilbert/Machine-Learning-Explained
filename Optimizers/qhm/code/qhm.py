# based on https://arxiv.org/pdf/1810.06801.pdf

import numpy as np


class QHM:
    """QHM -Quasi-Hyperbolic Momentum
    Parameters:
    -----------
    learning_rate: float = 0.001
        The step length used when following the negative gradient.
    beta: float = 0.999
        Momentum factor.
    v: float = 0.7
        Immediate discount factor.
    """
    def __init__(self, learning_rate: float = 0.001, beta: float = 0.999, v: float = 0.7) -> None:
        self.learning_rate = learning_rate
        self.beta = beta
        self.v = v

        self.g_t = np.array([])

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        if not self.g_t.any():
            self.g_t = np.zeros(np.shape(w))
        
        self.g_t = self.beta * self.g_t + (1 - self.beta) * grad_wrt_w

        return w - self.learning_rate * ((1 - self.v) * grad_wrt_w + self.v * self.g_t)
