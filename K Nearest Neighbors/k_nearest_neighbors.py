import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time


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
        return np.array([Counter(self.k_nearest(distances)).most_common()[0][0] for distances in distances_list])


if __name__ == '__main__':
    df = pd.read_csv('C:/Users/Gilbert/Desktop/Programming/Datasets/Iris/iris.csv')
    X, y =(np.array(df.drop('species', axis=1)), LabelEncoder().fit_transform(np.array(df['species'])))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNearestNeighbours(4)
    start = time.clock()
    model.fit(X_train, y_train)
    print(time.clock() - start)
    predictions = model.predict(X_test)
    print('Accuracy:', (predictions == y_test).sum()/len(predictions)*100)
