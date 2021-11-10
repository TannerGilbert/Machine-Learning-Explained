import numpy as np


class Mish:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x * np.tanh(np.log(1 + np.exp(x)))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return (np.exp(x) * (4*np.exp(2*x) + np.exp(3*x) + 4*(1+x) + np.exp(x)*(6+4*x))) / np.power(2 + 2*np.exp(x) + np.exp(2*x), 2)
