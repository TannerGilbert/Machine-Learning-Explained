import numpy as np


class MeanSquaredLogarithmicError:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return np.sum(np.power(np.log(1 + y) - np.log(1 + y_pred), 2)) / y.shape[0]
