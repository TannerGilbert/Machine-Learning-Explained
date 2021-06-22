from typing import Union
import numpy as np


class Softmax:
    def __call__(self, x: Union[list, np.ndarray]) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def gradient(self, x: Union[list, np.ndarray]) -> np.ndarray:
        p = self.__call__(x)
        return p * (1 - p)
