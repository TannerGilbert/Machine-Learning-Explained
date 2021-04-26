from __future__ import annotations
from typing import Union
import numpy as np


class KNearestNeighbors:

    def __init__(self, k: int) -> None:
        self.X = None
        self.y = None
        self.k = k

    def fit(self, X: Union[list, np.ndarray], y: Union[list, np.ndarray]) -> KNearestNeighbours:
        self.X = X
        self.y = y
        return self

    def euclidean_distance(self, X_test: Union[list, np.ndarray]) -> list:
        return [np.linalg.norm(X - X_test) for X in self.X]

    def k_nearest(self, X: Union[list, np.ndarray]) -> np.ndarray:
        idx = np.argpartition(X, self.k)
        return np.take(self.y, idx[:self.k])

    def predict(self, X: Union[list, np.ndarray]) -> np.ndarray:
        distances_list = [self.euclidean_distance(x) for x in X]
        return np.array([np.mean(self.k_nearest(distances)) for distances in distances_list])


if __name__ == '__main__':
    import pandas as pd
    iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
    iris = iris.sample(frac=1).reset_index(drop=True)
    X = np.array(iris.drop(['petal_width', 'label'], axis=1))
    y = np.array(iris['petal_width'])
    model = KNearestNeighbors(3)
    model.fit(X, y)
    print(model.predict(X[:5]))
    print(y[:5])
