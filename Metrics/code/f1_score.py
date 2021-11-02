import numpy as np
from recall import Recall
from precision import Precision


class F1Score:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        precision = Precision()
        recall = Recall()
        return 2 * (precision(y, y_pred) * recall(y, y_pred)) / (precision(y, y_pred) + recall(y, y_pred))