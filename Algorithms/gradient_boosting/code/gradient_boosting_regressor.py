# based on https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/gradient_boosting.py

from __future__ import annotations
from typing import Union
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def square_error_gradient(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return -(y - y_pred)


class GradientBoostingRegressor:
    """Gradient Boosting Regressor
    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    learning_rate: float
        The step length that will be taken when following the negative gradient.
    min_samples_split: int
        The minimum number of samples required to split an internal node.
    max_depth: int
        The maximum depth of the individual regression estimators..
    """
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, min_samples_split: int = 2,
                 max_depth: int = 3) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        # Initialize trees
        self.initial_prediction = None
        self.trees = []
        for _ in range(n_estimators):
            tree = DecisionTreeRegressor(min_samples_split=self.min_samples_split,
                                         max_depth=self.max_depth)
            self.trees.append(tree)

    def fit(self, X: Union[list, np.ndarray], y: np.ndarray) -> GradientBoostingRegressor:
        self.initial_prediction = np.mean(y, axis=0)
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))  # initial prediction
        for i in range(self.n_estimators):
            gradient = square_error_gradient(y, y_pred)
            self.trees[i].fit(X, gradient)
            update = self.trees[i].predict(X)
            # Update y predictions
            y_pred -= np.multiply(self.learning_rate, update)

        return self

    def predict(self, X: Union[list, np.ndarray]) -> np.ndarray:
        y_pred = np.array([])
        # Make predictions
        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_pred = self.initial_prediction - update if not y_pred.any() else y_pred - update
        return y_pred


if __name__ == '__main__':
    from sklearn import datasets
    # Load the diabetes dataset
    X, y = datasets.load_diabetes(return_X_y=True)
    model = GradientBoostingRegressor(max_depth=8)
    model.fit(X, y)
    print(model.predict(X[:5]))
    print(y[:5])
