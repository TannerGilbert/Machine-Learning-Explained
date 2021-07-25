from __future__ import annotations
import numpy as np


class NormalEquation:

    def __init__(self):
        self.w = None

    
    def fit(self, x: np.ndarray, y: np.ndarray) -> NormalEquation:
        x = np.append(np.ones([len(x), 1]), x, 1)
        z = np.linalg.inv(np.dot(x.transpose(), x))
        self.w = np.dot(np.dot(z, x.transpose()), y)
        return self

    def predict(self, x: np.ndarray):
        if self.w == None:
            raise Exception('Call .fit before using predict method')

        x = np.append(np.ones([len(x), 1]), x, 1)
        return np.dot(x, self.w)
