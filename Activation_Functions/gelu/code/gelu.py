import numpy as np
from scipy.special import erf


class GELU:
    def __call__(self, x: np.ndarray, approximate=True) -> np.ndarray:
        if approximate:
            return x / 2 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        return x / 2 * (1.0 + erf(x / np.sqrt(2.0)))

    # TODO: Add gradients
    def gradient(self, x: np.ndarray) -> np.ndarray:
        pass