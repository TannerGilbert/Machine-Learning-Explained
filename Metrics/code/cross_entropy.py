import numpy as np


class CrossEntropy:
    def __init__(self):
        self.epsilon = 1e-15

    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        # Avoid division by zero
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return - y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)

    def gradient(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        # Avoid division by zero
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return - (y / y_pred) + (1 - y) / (1 - y_pred)
