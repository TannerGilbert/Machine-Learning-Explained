from __future__ import annotations
from typing import Union
import numpy as np


class PCA:
    """PCA - Principal Component Analysis
    Parameters:
    -----------
    n_components: int = 2
        Number of components to keep.
    """
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None
        self.mean = None
        self.std = None

    def fit(self, X: Union[list, np.ndarray]) -> PCA:
        # subtract off the mean to center the data
        self.mean = X.mean(axis=0)
        X = X - self.mean

        covariance_matrix = np.cov(X, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:self.n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :self.n_components]

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

        return self

    def fit_transform(self, X: Union[list, np.ndarray]) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def transform(self, X: Union[list, np.ndarray]) -> np.ndarray:
        X = X - self.mean

        # Project the data onto principal components
        X_transformed = np.dot(X, self.eigenvectors)

        return X_transformed

    @property
    def explained_variance_ratio_(self):
        return self.eigenvalues / np.sum(self.eigenvalues)


if __name__ == '__main__':
    # source: https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn import datasets

    np.random.seed(5)

    centers = [[1, 1], [-1, -1], [1, -1]]
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
        ax.text3D(X[y == label, 0].mean(),
                X[y == label, 1].mean() + 1.5,
                X[y == label, 2].mean(), name,
                horizontalalignment='center',
                bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
            edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()