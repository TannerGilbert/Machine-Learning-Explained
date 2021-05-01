# based on https://ruder.io/optimizing-gradient-descent/#nesterovacceleratedgradient
# and https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/optimizers.py#L24

from typing import Callable
import numpy as np


class NesterovAcceleratedGradientDescent:
    """Gradient Descent with Nesterov Momentum
    Parameters:
    -----------
    learning_rate: float = 0.01
        The step length used when following the negative gradient.
    momentum: float = 0.0
        Amount of momentum to use.
        Momentum accelerates gradient descent in the relevant direction and dampens oscillations.
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.w_update = np.array([])

    def update(self, w: np.ndarray, grad_func: Callable) -> np.ndarray:
        # Calculate the gradient of the loss a bit further down the slope from w
        approx_future_grad = np.clip(grad_func(w - self.momentum * self.w_update), -1, 1)
        # Initialize w_update if not initialized yet
        if not self.w_update.any():
            self.w_update = np.zeros(np.shape(w))

        self.w_update = self.momentum * self.w_update + self.learning_rate * approx_future_grad
        # Move against the gradient to minimize loss
        return w - self.w_update
