from __future__ import annotations
from typing import Union
import numpy as np
import random


class KMeans:
    """KMeans
    Parameters:
    -----------
    n_clusters: int = 2
        Number of clusters
    tol: float = 0.0001
        Relative tolerance
    max_iter: int = 300
        Maximum number of iterations of the k-means algorithm for a single run
    """
    def __init__(self, n_clusters: int = 2, tol: float = 0.0001, max_iter: int = 300) -> None:
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = {}
        self.classifications = {}

    def fit(self, data: Union[list, np.ndarray]) -> KMeans:
        self.centroids = {}

        # Randomly initialize centroids
        for i in range(self.n_clusters):
            self.centroids[i] = data[random.randint(0, len(data)-1)]

        for _ in range(self.max_iter):
            self.classifications = {}

            for i in range(self.n_clusters):
                self.classifications[i] = []

            # put row into the right cluster
            for row in data:
                distances = [np.linalg.norm(
                    row-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(row)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(
                    self.classifications[classification], axis=0)

            # Check if we are finished
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    optimized = False

            if optimized:
                break
        return self

    def predict(self, data: Union[list, np.ndarray]) -> list:
        classifications = []
        for row in data:
            distances = [np.linalg.norm(row-self.centroids[centroid])
                         for centroid in self.centroids]
            classification = distances.index(min(distances))
            classifications.append(classification)
        return classifications


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import style
    from sklearn.datasets import make_blobs
    style.use('ggplot')

    X, y = make_blobs(n_samples=30, centers=3, n_features=2)

    model = KMeans(n_clusters=3)
    model.fit(X)

    centroids = model.centroids

    colors = 10 * ['r', 'g', 'b', 'c', 'k', 'y']

    for classification, featureset in zip(model.predict(X), X):
        color = colors[classification]
        plt.scatter(featureset[0], featureset[1], marker="x",
                    color=color, s=150, linewidths=5, zorder=10)

    for c in centroids:
        plt.scatter(centroids[c][0], centroids[c][1],
                    color='k', marker="*", s=150, linewidths=5)

    plt.show()