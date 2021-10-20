import numpy as np


class MeanAbsolutePercentageError:
    def __init__(self, eps: float = 1e-07):
        self.eps = eps

    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return np.sum(np.absolute((y - y_pred)) / np.maximum(self.eps, np.absolute(y))) / y.shape[0]
