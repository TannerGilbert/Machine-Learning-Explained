import numpy as np
from sklearn.tree import DecisionTreeRegressor


class RandomForest:
    def __init__(self, n_estimators=10, n_features='sqrt', sample_size=0.8, max_depth=10, min_leaf=5):
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.sample_size = sample_size
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            idxs = np.random.permutation(len(X))[:int(self.sample_size*len(X))]

            self.trees.append(DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_leaf=self.min_leaf, max_features=self.n_features).fit(X[idxs], y[idxs]))
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self.trees], axis=0)
 