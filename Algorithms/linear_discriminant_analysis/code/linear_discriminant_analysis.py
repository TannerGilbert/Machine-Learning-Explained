# based on https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/multi_class_lda.py

from __future__ import annotations
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt


class LDA:
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, X, y):
        if self.n_components is None or self.n_components > X.shape[1]:
            n_components = X.shape[1]
        else:
            n_components = self.n_components
        
        n_features = np.shape(X)[1]
        labels = np.unique(y)
        
        # Within class scatter matrix
        S_W = np.empty((n_features, n_features))
        for label in labels:
            _X = X[y == label]
            S_W += (len(_X) - 1) * np.cov(_X, rowvar=False)


        # Between class scatter matrix
        total_mean = np.mean(X, axis=0)
        S_B = np.empty((n_features, n_features))
        for label in labels:
            _X = X[y == label]
            _mean = np.mean(_X, axis=0)
            S_B += len(_X) * (_mean - total_mean).dot((_mean - total_mean).T)
            
        # Determine SW^-1 * SB by calculating inverse of SW
        A = np.linalg.inv(S_W).dot(S_B)

        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eigh(A)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = eigenvectors[:, idx][:, :n_components]
        
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
    def fit_transform(self, X: Union[list, np.ndarray]) -> np.ndarray:
        self.fit(X)
        return self.transform(X)        
    
    def transform(self, X):
        return np.dot(X, self.eigenvectors)
    
    @property
    def explained_variance_ratio_(self) -> np.ndarray:
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
    pca = LDA(n_components=3)
    pca.fit(X, y)
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