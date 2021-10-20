# based on https://keras.io/api/losses/probabilistic_losses/#poisson-class
import numpy as np


class Poisson:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return np.sum(y_pred - y * np.log(y_pred)) / y.shape[0]
