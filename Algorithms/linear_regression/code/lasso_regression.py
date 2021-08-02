from __future__ import annotations
from typing import Tuple
import numpy as np


class LassoRegression:
    """Lasso Regression
    Parameters:
    -----------
    learning_rate: float
        The step length used when following the negative gradient during training.
    C: float, default=1
       Regularization strength
    """
    def __init__(self, learning_rate: float, C: float = 1) -> None:
        self.learning_rate = learning_rate
        self.C = C
        self.w = ""

    def cost_function(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        dif = np.dot(x, self.w) - y
        cost = (np.sum(dif**2) + self.C * np.sum(np.absolute(self.w))) / (2*np.shape(x)[0])

        return dif, cost

    def fit(self, x: np.ndarray, y: np.ndarray, num_iterations: int = 10000) -> LassoRegression:
        if self.w == "":
            _, num_features = np.shape(x)
            self.w = np.random.uniform(-1, 1, num_features)
        for _ in range(num_iterations):
            dif, cost = self.cost_function(x, y)
            gradient = np.dot(x.transpose(), dif) / np.shape(x)[0]
            self.w = self.w - self.learning_rate * gradient
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.w)


# Testing functionality
if __name__ == '__main__':
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
    le = LabelEncoder()
    iris['label'] = le.fit_transform(iris['label'])
    X = np.array(iris.drop(['petal_width'], axis=1))
    y = np.array(iris['petal_width'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LassoRegression(0.0001)
    model.fit(X_train, y_train, 10000)
    predictions = model.predict(X_test)
    mse = ((y_test - predictions)**2).mean(axis=0)
    print('Loss:', mse)
