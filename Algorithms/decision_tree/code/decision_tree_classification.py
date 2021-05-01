from __future__ import annotations
from typing import Union, Optional
import numpy as np
import pandas as pd
from collections import Counter


def gini_index(groups_x: Union[list, np.ndarray], groups_y: Union[list, np.ndarray], classes: Union[list, np.ndarray]) -> float:
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups_x]))
    # sum weighted Gini index for each group
    gini = 0.0
    for i in range(len(groups_x)):
        size = float(len(groups_x[i]))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [y for y in groups_y[i]].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


class DecisionTree:
    """Decision Tree Classifier
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
    classes: Union[list, np.ndarray]
    depth: Optional[int] = 10
        The maximum depth of the tree.
    min_leaf: Union[int, float] = 5
        The minimum number of samples required to be at a leaf node.
    """
    def __init__(self, x: Union[pd.DataFrame, pd.Series], y: Union[pd.DataFrame, pd.Series], n_features: int, f_idxs: Union[list, np.ndarray], idxs: Union[list, np.ndarray], classes: Union[list, np.ndarray], depth: Optional[int] = 10, min_leaf: Optional[int] = 5) -> None:
        self.x, self.y, self.idxs, self.min_leaf, self.f_idxs = x, y, idxs, min_leaf, f_idxs
        self.depth = depth
        self.n_features = n_features
        self.n, self.c = len(idxs), x.shape[1]
        self.classes = classes
        self.score = float('inf')
        self.val = Counter(y[idxs]).most_common()[0][0]
        self.find_varsplit()

    def find_varsplit(self) -> None:
        for i in self.f_idxs:
            self.find_better_split(i)
        if self.is_leaf:
            return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        self.lhs = DecisionTree(self.x, self.y, self.n_features, lf_idxs, self.idxs[lhs], self.classes, depth=self.depth - 1,
                                min_leaf=self.min_leaf)
        self.rhs = DecisionTree(self.x, self.y, self.n_features, rf_idxs, self.idxs[rhs], self.classes, depth=self.depth - 1,
                                min_leaf=self.min_leaf)

    def find_better_split(self, var_idx: int) -> None:
        x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y, sort_x = y[sort_idx], x[sort_idx]
        for i in range(1, self.n - 1):
            if i < self.min_leaf or sort_x[i] == sort_x[i + 1]:
                continue
            x_l, x_r = sort_x[:i], sort_x[i:]
            y_l, y_r = sort_y[:i], sort_y[i:]

            curr_score = gini_index([x_l, x_r], [y_l, y_r], self.classes)
            if curr_score < self.score:
                self.var_idx, self.score, self.split = var_idx, curr_score, sort_x[i]

    @property
    def split_col(self) -> list:
        return self.x.values[self.idxs, self.var_idx]

    @property
    def is_leaf(self) -> bool:
        return self.score == float('inf') or self.depth <= 0

    def predict(self, x: Union[list, np.ndarray]) -> np.ndarray:
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi: Union[list, np.ndarray]) -> int:
        if self.is_leaf:
            return self.val
        t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return t.predict_row(xi)
