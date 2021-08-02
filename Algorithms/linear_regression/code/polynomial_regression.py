from __future__ import annotations
from typing import Tuple
from itertools import combinations_with_replacement
import numpy as np


# from https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/utils/data_manipulation.py#L43
def polynomial_features(X: np.ndarray, degree: float) -> np.ndarray:
    n_samples, n_features = np.shape(X)

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    
    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))
    
    for i, index_combs in enumerate(combinations):  
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new


class PolynomialRegression:
    """Polynomial Regression
    Parameters:
    -----------
    learning_rate: float
        The step length used when following the negative gradient during training.
    """
    def __init__(self, learning_rate: float, degree: float = 2) -> None:
        self.learning_rate = learning_rate
        self.degree = degree
        self.w = ""

    def cost_function(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        dif = np.dot(x, self.w) - y
        cost = np.sum(dif**2) / (2*np.shape(x)[0])

        return dif, cost

    def fit(self, x: np.ndarray, y: np.ndarray, num_iterations: int = 10000) -> PolynomialRegression:
        x = polynomial_features(x, self.degree)
        if self.w == "":
            _, num_features = np.shape(x)
            self.w = np.random.uniform(-1, 1, num_features)
        for i in range(num_iterations):
            dif, cost = self.cost_function(x, y)
            gradient = np.dot(x.transpose(), dif) / np.shape(x)[0]
            self.w = self.w - self.learning_rate * gradient
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = polynomial_features(x, self.degree)
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

    model = PolynomialRegression(0.0001)
    model.fit(X_train, y_train, 10000)
    predictions = model.predict(X_test)
    mse = ((y_test - predictions)**2).mean(axis=0)
    print('Loss:', mse)
