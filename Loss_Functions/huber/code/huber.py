import numpy as np


class Huber:
    def __init__(self, delta: float = 1.) -> None:
        self.delta = delta

    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return np.where(np.abs(y - y_pred) < self.delta, 0.5 * (y - y_pred)**2, self.delta * (np.abs(y - y_pred)- 0.5 * self.delta))
