import numpy as np


class AdamW:
    """AdamW
    Parameters:
    -----------
    learning_rate: float = 0.001
        The step length used when following the negative gradient.
    beta_1: float = 0.9
        The exponential decay rate for the 1st moment estimates.
    beta_2: float = 0.999
        The exponential decay rate for the 2nd moment estimates.
    epsilon: float = 1e-07
        A small floating point value to avoid zero denominator.
    """
    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-7, weight_decay: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay

        self.t = 0
        self.m = None  # Decaying averages of past gradients
        self.v = None  # Decaying averages of past squared gradients

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.m is None:
            self.m = np.zeros(np.shape(grad_wrt_w))
            self.v = np.zeros(np.shape(grad_wrt_w))

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_wrt_w
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(grad_wrt_w, 2)

        m_hat = self.m / (1 - self.beta_1**self.t)
        v_hat = self.v / (1 - self.beta_2**self.t)

        w_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * grad_wrt_w

        return w - w_update
