import numpy as np


class TanH:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.power(self.__call__(x), 2)