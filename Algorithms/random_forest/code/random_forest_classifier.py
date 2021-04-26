from __future__ import annotations
from typing import Union, Optional
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class RandomForest:
    """Random Forest Classifier
    Parameters:
    -----------
    n_estimators: int = 10
        The number of trees in the forest.
    n_features: Optional[Union[str, int]] = 'sqrt'
        The number of features to consider when looking for the best split
    sample_size: float = 0.8
        Amount of data used (0-1)
    max_depth: Optional[int] = 10
        The maximum depth of the tree.
    min_leaf: Union[int, float] = 5
        The minimum number of samples required to be at a leaf node.
    """
    def __init__(self, n_estimators: int = 10, n_features: Optional[Union[str, int]] = 'sqrt', sample_size: float = 0.8,
                 max_depth: Optional[int] = 10, min_leaf: Union[int, float] = 5) -> None:
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.sample_size = sample_size
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.trees = []

    def fit(self, X: Union[list, np.ndarray], y: Union[list, np.ndarray]) -> RandomForest:
        for _ in range(self.n_estimators):
            idxs = np.random.permutation(len(X))[:int(self.sample_size*len(X))]

            self.trees.append(DecisionTreeClassifier(
                max_depth=self.max_depth, min_samples_leaf=self.min_leaf, max_features=self.n_features).fit(X[idxs], y[idxs]))
        return self

    def predict(self, X: Union[list, np.ndarray]) -> np.ndarray:
        predictions_array = np.column_stack([t.predict(X) for t in self.trees])
        return np.array([np.argmax(np.bincount(predictions)) for predictions in predictions_array])
