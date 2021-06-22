import numpy as np


class SoftPlus:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
