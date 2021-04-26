from __future__ import annotations
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))


class LogisticRegression:
    """Logistic Regression
    Parameters:
    -----------
    learning_rate: float
        The step length used when following the negative gradient during training.
    num_features: int
        The number of feature in the data
    penalty: str, default='l2'
        The type of penalty used.
    C: float, default=1
       Regularization strength
    """
    def __init__(self, learning_rate: float, num_features: int, penalty: str = 'l2', C: float = 0.1) -> None:
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.C = C
        self.b = 0
        self.w = np.zeros((1, num_features))
        assert penalty in ['l2', 'l1', None]

    def cost_function(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        y_T = y.T
        if self.penalty == 'l1':
            return (-1/y.shape[0]) * (np.sum((y_T*np.log(y_pred)) + ((1-y_T) * np.log(1-y_pred))) + self.C * np.sum(np.absolute(self.w)))
        elif self.penalty == 'l2':
            return (-1/y.shape[0]) * (np.sum((y_T*np.log(y_pred)) + ((1-y_T) * np.log(1-y_pred))) + self.C * np.sum(np.square(self.w)))
        else:
            return (-1/y.shape[0]) * (np.sum((y_T*np.log(y_pred)) + ((1-y_T) * np.log(1-y_pred))))

    def fit(self, X: np.ndarray, y: np.ndarray, num_iterations) -> LogisticRegression:
        for i in range(num_iterations):
            pred = sigmoid(np.dot(self.w, X.T) + self.b)
            cost = self.cost_function(y, pred)

            # Calculate Gradients/Derivatives
            dw = (1 / X.shape[0]) * (np.dot(X.T, (pred - y.T).T))
            db = (1 / X.shape[0]) * (np.sum(pred - y.T))

            self.w = self.w - (self.learning_rate * dw.T)
            self.b = self.b - (self.learning_rate * db)
        return self

    def predict(self, X: np.ndarray) -> list:
        predictions = sigmoid(np.dot(self.w, X.T) + self.b)[0]
        return [1 if pred >= 0.5 else 0 for pred in predictions]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(np.dot(self.w, X.T) + self.b)[0]

