import numpy as np


class KNearestNeighbours:

    def __init__(self, k: int) -> None:
        self.X = None
        self.y = None
        self.k = k

    def fit(self, X: np.array, y: np.array) -> None:
        self.X = X
        self.y = y

    def euclidean_distance(self, X_test):
        return [np.linalg.norm(X - X_test) for X in self.X]

    def k_nearest(self, X) -> np.array:
        idx = np.argpartition(X, self.k)
        return np.take(self.y, idx[:self.k])

    def predict(self, X: np.array) -> np.array:
        distances_list = [self.euclidean_distance(x) for x in X]
        return np.array([np.mean(self.k_nearest(distances)) for distances in distances_list])


if __name__ == '__main__':
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
    iris = iris.sample(frac=1).reset_index(drop=True)
    le = LabelEncoder()
    iris['label'] = le.fit_transform(iris['label'])
    X = np.array(iris.drop(['petal_width'], axis=1))
    y = np.array(iris['petal_width'])
    model = KNearestNeighbours(3)
    model.fit(X, y)
    print(model.predict(X[:3]))
    print(y[:3])
