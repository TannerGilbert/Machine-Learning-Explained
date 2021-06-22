import numpy as np


class SiLU:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x / (1 + np.exp(-x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return (1 + np.exp(-x) + x * np.exp(-x)) / np.power(1 + np.exp(-x), 2)
