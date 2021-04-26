from __future__ import annotations
import numpy as np
from logistic_regression import LogisticRegression


class LogisticRegressionOneVsAll:
    """One vs. All Logistic Regression
    Parameters:
    -----------
    learning_rate: float
        The step length used when following the negative gradient during training.
    num_features: int
        The number of feature in the data
    num_classes: int
        The number of classes in the data-set
    """
    def __init__(self, learning_rate: float, num_features: int, num_classes: int) -> None:
        self.models = [LogisticRegression(learning_rate, num_features) for _ in range(num_classes)]

    def fit(self, X: np.ndarray, y: np.ndarray, num_iterations: int) -> LogisticRegressionOneVsAll:
        for i, model in enumerate(self.models):
            y_tmp = (y == i).astype(int)
            model.fit(X, y_tmp, num_iterations)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([model.predict_proba(X) for model in self.models])
        return np.argmax(predictions, axis=0)
