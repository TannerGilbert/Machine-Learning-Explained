# based on https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/gradient_boosting.py

from __future__ import annotations
from typing import Union
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def cross_entropy_gradient(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return - (y / p) + (1 - y) / (1 - p)


class GradientBoostingClassifier:
    """Gradient Boosting Classifier
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

    def fit(self, X: Union[list, np.ndarray], y: np.ndarray) -> GradientBoostingClassifier:
        # one-hot encode y
        y_one_hot = np.zeros((y.shape[0], np.amax(y) + 1))
        y_one_hot[np.arange(y.shape[0]), y] = 1
        y = y_one_hot

        self.initial_prediction = np.mean(y, axis=0)
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))  # initial prediction
        for i in range(self.n_estimators):
            gradient = cross_entropy_gradient(y, y_pred)
            self.trees[i].fit(X, gradient)
            update = self.trees[i].predict(X)
            # Update y predictions
            y_pred -= np.multiply(self.learning_rate, update)

        return self

    def predict(self, X: Union[list, np.ndarray]) -> np.ndarray:
        y_pred = np.array([])
        for tree in self.trees:
            update = np.multiply(self.learning_rate, tree.predict(X))
            y_pred = self.initial_prediction - update if not y_pred.any() else y_pred - update

        # Turn y_pred intro probability distribution
        y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                     names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
    X, y = (np.array(df.drop('label', axis=1)),
            LabelEncoder().fit_transform(np.array(df['label'])))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('Accuracy:', (predictions == y_test).sum()/len(predictions)*100)
