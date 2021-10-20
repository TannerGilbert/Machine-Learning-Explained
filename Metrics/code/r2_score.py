import numpy as np


class R2Score:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return 1 - (np.sum(np.power(y-y_pred, 2))) / (np.sum(np.power(y-np.mean(y), 2)))
