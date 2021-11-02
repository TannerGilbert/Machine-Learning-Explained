# based on https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co

from sklearn.metrics import confusion_matrix
import numpy as np


class Recall:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        cm = confusion_matrix(y, y_pred)
        return np.mean(np.diag(cm) / np.sum(cm, axis=1))