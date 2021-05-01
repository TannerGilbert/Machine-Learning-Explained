# based on https://ruder.io/optimizing-gradient-descent/#adagrad
# and https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/optimizers.py#L41

import numpy as np


class Adagrad:
    """Adagrad
    Parameters:
    -----------
    learning_rate: float = 0.001
        The step length used when following the negative gradient.
    initial_accumulator_value: float = 0.1
        Starting value for the accumulators, must be non-negative.
    epsilon: float = 1e-07
        A small floating point value to avoid zero denominator.
    """
    def __init__(self, learning_rate: float = 0.001, initial_accumulator_value: float = 0.1, epsilon: float = 1e-07) -> None:
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon
        self.G = np.array([])  # Sum of squares of the gradients

        assert self.initial_accumulator_value > 0, "initial_accumulator_value must be non-negative"

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        # Initialize w_update if not initialized yet
        if not self.G.any():
            self.G = np.full(np.shape(w), self.initial_accumulator_value)
        # Add the square of the gradient of the loss function at w
        self.G += np.power(grad_wrt_w, 2)
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.G + self.epsilon)
