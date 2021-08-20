import numpy as np


class Hinge:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum(np.maximum(0, 1 - y * y_pred)) / len(y)
