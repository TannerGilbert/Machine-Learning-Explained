import numpy as np
from scipy.special import erf


class GELU:
    def __call__(self, x: np.ndarray, approximate: bool = True) -> np.ndarray:
        if approximate:
            return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

    def gradient(self, x: np.ndarray, approximate: bool = True) -> np.ndarray:
        if approximate:
            return 0.5 * np.tanh(0.0356774 * np.power(x, 3) + 0.797885 * x) + (0.0535161 * np.power(x, 3) + 0.398942 * x) * np.power(1 / np.cosh(x), 2) * (0.0356774 * np.power(x, 3) + 0.797885 * x) + 0.5
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0))) + x * 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * np.power(x, 2))
