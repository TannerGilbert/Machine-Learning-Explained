import numpy as np


class ELU:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1.0))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0.0, 1.0, self.alpha * np.exp(x))
