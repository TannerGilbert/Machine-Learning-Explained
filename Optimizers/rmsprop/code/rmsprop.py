# based on https://ruder.io/optimizing-gradient-descent/#rmsprop
# and https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/optimizers.py#L88

import numpy as np


class RMSprop:
    """RMSprop
    Parameters:
    -----------
    learning_rate: float = 0.001
        The step length used when following the negative gradient.
    rho: float = 0.9
        Discounting factor for the history/coming gradient.
    epsilon: float = 1e-07
        A small floating point value to avoid zero denominator.
    """
    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9, epsilon: float = 1e-7) -> None:
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.E_grad = None  # Running average of the square gradients at w

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        if self.E_grad is None:
            self.E_grad = np.zeros(np.shape(grad_wrt_w))

        # Update average of gradients at w
        self.E_grad = self.rho * self.E_grad + (1 - self.rho) * np.power(grad_wrt_w, 2)

        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.E_grad + self.epsilon)
