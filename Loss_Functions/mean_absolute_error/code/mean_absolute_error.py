import numpy as np


class MeanAbsoluteError:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return 0.5 * np.sum(np.absolute(y - y_pred)) / y.shape[0]
