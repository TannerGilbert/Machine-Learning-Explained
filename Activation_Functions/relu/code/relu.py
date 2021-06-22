from typing import Union
import numpy as np


class ReLU:
    def __call__(self, x: Union[list, np.ndarray]) -> np.ndarray:
        return np.maximum(x, 0.0)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0.0, 1.0, 0.0)
