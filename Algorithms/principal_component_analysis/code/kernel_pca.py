# based on https://sebastianraschka.com/Articles/2014_kernel_pca.html

from __future__ import annotations
from typing import Union
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh


class KernelPCA:
    """KernelPCA
    Parameters:
    -----------
    n_components: int = 2
        Number of components to keep.
    gamma: float = None
        Kernel coefficient
    """
    def __init__(self, n_components: int = 2, gamma: float = None):
        self.n_components = n_components
        self.gamma = gamma
        self.alphas = None
        self.lambdas = None
        self.X = None

    def fit(self, X: Union[list, np.ndarray]) -> KernelPCA:
        if self.gamma == None:
            self.gamma = 1 / X.shape[1]

        sq_dists = pdist(X, 'sqeuclidean')

        mat_sq_dists = squareform(sq_dists)

        K = np.exp(-self.gamma * mat_sq_dists)

        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K_norm = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        eigenvalues, eigenvectors = eigh(K_norm)

        alphas = np.column_stack((eigenvectors[:,-i] for i in range(1, self.n_components+1)))
        lambdas = [eigenvalues[-i] for i in range(1, self.n_components+1)]

        self.alphas = alphas
        self.lambdas = lambdas
        self.X = X

        return self

    def fit_transform(self, X: Union[list, np.ndarray]) -> np.ndarray:
        self.fit(X)
        return self.alphas * np.sqrt(self.lambdas)

    def transform(self, X: Union[list, np.ndarray]) -> np.ndarray:
        # TODO: Rewrite as this is very inefficient
        def transform_row(X_r):
            pair_dist = np.array([np.sum((X_r-row)**2) for row in self.X])
            k = np.exp(-self.gamma * pair_dist)
            return k.dot(self.alphas / self.lambdas)
        
        return np.array(list(map(transform_row, X)))
        



if __name__ == '__main__':
    from sklearn.datasets import make_circles
    import matplotlib.pyplot as plt

    X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

    plt.figure(figsize=(8,6))

    pca = KernelPCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    print(X)