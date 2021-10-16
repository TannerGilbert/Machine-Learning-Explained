import numpy as np


class MeanSquaredError:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return 0.5 * np.linalg.norm(y_pred - y) ** 2 / y.shape[0]

    def gradient(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return np.linalg.norm(y_pred - y) / y.shape[0]
