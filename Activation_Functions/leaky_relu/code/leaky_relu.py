import numpy as np


class LeakyReLU:
    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0.0, x, self.alpha * x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0.0, 1.0, self.alpha)
