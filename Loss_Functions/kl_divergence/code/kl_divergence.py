import numpy as np


class KLDivergence:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return np.sum(np.where(y != 0, y * np.log(y / y_pred), 0))