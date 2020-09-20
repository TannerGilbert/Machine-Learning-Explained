import numpy as np
import pandas as pd
from collections import Counter


def gini_index(groups_x, groups_y, classes):
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
    def __init__(self, x, y, n_features, f_idxs, idxs, classes, depth=10, min_leaf=5):
        self.x, self.y, self.idxs, self.min_leaf, self.f_idxs = x, y, idxs, min_leaf, f_idxs
        self.depth = depth
        self.n_features = n_features
        self.n, self.c = len(idxs), x.shape[1]
        self.classes = classes
        self.score = float('inf')
        self.val = Counter(y[idxs]).most_common()[0][0]
        self.find_varsplit()

    def find_varsplit(self):
        for i in self.f_idxs: self.find_better_split(i)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        self.lhs = DecisionTree(self.x, self.y, self.n_features, lf_idxs, self.idxs[lhs], self.classes, depth=self.depth - 1,
                                min_leaf=self.min_leaf)
        self.rhs = DecisionTree(self.x, self.y, self.n_features, rf_idxs, self.idxs[rhs], self.classes, depth=self.depth - 1,
                                min_leaf=self.min_leaf)

    def find_better_split(self, var_idx):
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
    def split_name(self):
        return self.x.columns[self.var_idx]

    @property
    def split_col(self):
        return self.x.values[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        return self.score == float('inf') or self.depth <= 0

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return t.predict_row(xi)


if __name__ == '__main__':
    df = pd.read_csv('D:/Datasets/Heart Disease UCI/heart.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    X, y = [df.drop('target', axis=1), np.array(df['target'])]
    model = DecisionTree(X, y, X.shape[1], np.array([i for i in range(X.shape[1])]), np.array([i for i in range(X.shape[0])]), [0, 1])
    print(model.predict(np.array(X)[:5]))
    print(y[:5])
