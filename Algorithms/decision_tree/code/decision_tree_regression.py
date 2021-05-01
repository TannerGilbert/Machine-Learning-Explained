from __future__ import annotations
from typing import Union, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math


def std_agg(cnt: float, s1: float, s2: float) -> float:
    return math.sqrt(abs((s2/cnt) - (s1/cnt)**2))


class DecisionTree:
    """Decision Tree Regressor
    Parameters:
    -----------
    x: Union[pd.DataFrame, pd.Series]
        The input data
    y: Union[pd.DataFrame, pd.Series]
        The output
    n_features: int
        The number of features to consider when looking for the best split
    f_idxs: Union[list, np.ndarray]
        The indexes of the selected features
    idxs: Union[list, np.ndarray]
        The indexes of the data-points inside the data-set to use for the tree
    depth: Optional[int] = 10
        The maximum depth of the tree.
    min_leaf: Union[int, float] = 5
        The minimum number of samples required to be at a leaf node.
    """

    def __init__(self, x: Union[pd.DataFrame, pd.Series], y: Union[pd.DataFrame, pd.Series], n_features: int,
                 f_idxs: Union[list, np.ndarray], idxs: Union[list, np.ndarray], depth: Optional[int] = 10,
                 min_leaf: Optional[int] = 5) -> None:
        self.x, self.y, self.idxs, self.min_leaf, self.f_idxs = x, y, idxs, min_leaf, f_idxs
        self.depth = depth
        self.n_features = n_features
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()

    def find_varsplit(self) -> None:
        for i in self.f_idxs: self.find_better_split(i)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        self.lhs = DecisionTree(self.x, self.y, self.n_features, lf_idxs, self.idxs[lhs], depth=self.depth - 1,
                                min_leaf=self.min_leaf)
        self.rhs = DecisionTree(self.x, self.y, self.n_features, rf_idxs, self.idxs[rhs], depth=self.depth - 1,
                                min_leaf=self.min_leaf)

    def find_better_split(self, var_idx: int) -> None:
        x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y, sort_x = y[sort_idx], x[sort_idx]
        rhs_cnt, rhs_sum, rhs_sum2 = self.n, sort_y.sum(), (sort_y ** 2).sum()
        lhs_cnt, lhs_sum, lhs_sum2 = 0, 0., 0.

        for i in range(0, self.n - self.min_leaf - 1):
            xi, yi = sort_x[i], sort_y[i]
            lhs_cnt += 1;
            rhs_cnt -= 1
            lhs_sum += yi;
            rhs_sum -= yi
            lhs_sum2 += yi ** 2;
            rhs_sum2 -= yi ** 2
            if i < self.min_leaf or xi == sort_x[i + 1]:
                continue

            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std * lhs_cnt + rhs_std * rhs_cnt
            if curr_score < self.score:
                self.var_idx, self.score, self.split = var_idx, curr_score, xi

    @property
    def split_col(self):
        return self.x.values[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        return self.score == float('inf') or self.depth <= 0

    def predict(self, x: Union[list, np.ndarray]) -> np.ndarray:
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi: Union[list, np.ndarray]) -> int:
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return t.predict_row(xi)


if __name__ == '__main__':
    iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
    iris = iris.sample(frac=1).reset_index(drop=True)
    le = LabelEncoder()
    iris['label'] = le.fit_transform(iris['label'])
    X = iris.drop(['petal_width'], axis=1)
    y = np.array(iris['petal_width'])
    model = DecisionTree(X, y, X.shape[1], np.array([i for i in range(X.shape[1])]), np.array([i for i in range(X.shape[0])]))
    print(model.predict(np.array(X)[:5]))
    print(y[:5])
