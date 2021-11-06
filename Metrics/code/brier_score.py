import numpy as np


class BrierScore:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return np.sum(np.power(y - y_pred, 2)) / y.shape[0]
