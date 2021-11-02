import numpy as np
from recall import Recall
from precision import Precision


class FBetaScore:
    def __init__(self, beta: float = 1.) -> None:
        self.beta = beta

    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        precision = Precision()
        recall = Recall()
        return (1 + pow(self.beta, 2)) * (precision(y, y_pred) * recall(y, y_pred)) / ((pow(self.beta, 2) * precision(y, y_pred)) + recall(y, y_pred))