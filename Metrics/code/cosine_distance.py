import numpy as np


class CosineDistance:

    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return np.dot(y, y_pred) / (np.linalg.norm(y) * np.linalg.norm(y_pred))
