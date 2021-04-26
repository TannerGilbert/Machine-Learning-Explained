# based on https://geoffruddock.com/adaboost-from-scratch-in-python/

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    def __init__(self, n_estimators: int):
        self.n_estimators = n_estimators
        self.stumps = np.zeros(shape=n_estimators, dtype=object)
        self.stump_weights = np.zeros(shape=n_estimators)
        self.sample_weights = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        self.sample_weights = np.zeros(shape=(self.n_estimators, n))

        # Initialize weights
        self.sample_weights[0] = np.ones(shape=n) / n

        for i in range(self.n_estimators):
            # fit weak learner
            curr_sample_weights = self.sample_weights[i]
            stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            stump.fit(X, y, sample_weight=curr_sample_weights)

            # calculate error and stump weight
            pred = stump.predict(X)
            err = curr_sample_weights[(pred != y)].sum()
            stump_weight = np.log((1 - err) / err) / 2

            # update sample weights
            new_sample_weights = (
                curr_sample_weights * np.exp(-stump_weight * y * pred)
            )

            # normalize sample weights
            new_sample_weights /= new_sample_weights.sum()

            if i+1 < self.n_estimators:
                self.sample_weights[i+1] = new_sample_weights

            self.stumps[i] = stump
            self.stump_weights[i] = stump_weight
        
        return self

    def predict(self, X):
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        return np.sign(np.dot(self.stump_weights, stump_preds))
