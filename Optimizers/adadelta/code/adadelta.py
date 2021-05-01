# based on https://ruder.io/optimizing-gradient-descent/#adadelta
# and https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/optimizers.py#L56

import numpy as np


class Adadelta:
    """Adadelta
    Parameters:
    -----------
    rho: float = 0.95
        The decay rate.
    epsilon: float = 1e-07
        A small floating point value to avoid zero denominator.
    """
    def __init__(self, rho: float = 0.95, epsilon: float = 1e-7) -> None:
        self.E_w_update = None  # Running average of squared parameter updates
        self.E_grad = None    # Running average of the squared gradient of w
        self.w_update = None    # Parameter update
        self.epsilon = epsilon
        self.rho = rho

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        if self.w_update is None:
            self.w_update = np.zeros(np.shape(w))
            self.E_w_update = np.zeros(np.shape(w))
            self.E_grad = np.zeros(np.shape(grad_wrt_w))

        # Update average of gradients at w
        self.E_grad = self.rho * self.E_grad + (1 - self.rho) * np.power(grad_wrt_w, 2)

        # Calculate root mean squared error of the weight update and gradients
        RMS_delta_w = np.sqrt(self.E_w_update + self.epsilon)
        RMS_grad = np.sqrt(self.E_grad + self.epsilon)

        # Calculate adaptive learning rate
        adaptive_lr = RMS_delta_w / RMS_grad

        # Calculate the update
        self.w_update = adaptive_lr * grad_wrt_w

        # Update the running average of w updates
        self.E_w_update = self.rho * self.E_w_update + (1 - self.rho) * np.power(self.w_update, 2)

        return w - self.w_update
