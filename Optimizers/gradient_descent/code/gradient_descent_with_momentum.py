# based on https://ruder.io/optimizing-gradient-descent/
# and https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/optimizers.py#L9

import numpy as np


class GradientDescent:
    """Gradient Descent with Momentum
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

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        # Initialize w_update if not initialized yet
        if not self.w_update.any():
            self.w_update = np.zeros(np.shape(w))
        # Use momentum if set
        self.w_update = self.momentum * self.w_update + (1 - self.momentum) * grad_wrt_w
        # Move against the gradient to minimize loss
        return w - self.learning_rate * self.w_update
