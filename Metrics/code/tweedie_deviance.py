import numpy as np


class TweedieDeviance:
    def __init__(self, power: int) -> None:
        self.power = power

    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        if self.power == 0:
            return np.sum(np.power(y - y_pred, 2)) / y.shape[0]
        elif self.power == 1:
            return np.sum(2 * (y * np.log(y / y_pred) + y_pred - y)) / y.shape[0]
        elif self.power == 2:
            return np.sum(2 * (np.log(y_pred / y) + y / y_pred - 1)) / y.shape[0]
        else:
            return np.sum(2 * (np.power(np.maximum(y, 0), 2-self.power) / ((1-self.power) * (2-self.power)) - (y * np.power(y_pred, 1 - self.power)) / (1 - self.power) + np.power(y_pred, 2 - self.power) / (2 - self.power))) / y.shape[0]
