import numpy as np


class Sigmoid:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.__call__(x) * (1 - self.__call__(x))
