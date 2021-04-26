from typing import Union
import numpy as np


class DBSCAN:
    """DBSCAN
    Parameters:
    -----------
    eps: float = 0.3
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_points: int = 5
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    """
    def __init__(self, eps: float = 0.3, min_points: int = 5) -> None:
        self.eps = eps
        self.min_points = min_points
        self.labels = []
        self.c = 1  # number of clusters

    def fit_predict(self, data: Union[list, np.ndarray]) -> list:
        self.labels = [0] * len(data)
        for i in range(len(data)):
            if not (self.labels[i] == 0):
                continue

            neighbours = self.find_neighbours(data, i)

            # If the number of points is below min_points the point is a outlier
            if len(neighbours) < self.min_points:
                self.labels[i] = -1
            else:
                self.grow_cluster(data, i, neighbours)
                self.c += 1
        return self.labels

    def find_neighbours(self, data: Union[list, np.ndarray], index: int) -> list:
        neighbors = []

        for p in range(len(data)):
            if np.linalg.norm(data[index]-data[p]) < self.eps and index != p:
                neighbors.append(p)
        return neighbors

    def grow_cluster(self, data: Union[list, np.ndarray], index: int, neighbours: list) -> None:
        # Assign seed point to cluster
        self.labels[index] = self.c

        i = 0
        while i < len(neighbours):
            p = neighbours[i]
            if self.labels[p] == -1:
                self.labels[p] = self.c
            elif self.labels[p] == 0:
                self.labels[p] = self.c
                neighbours_new = self.find_neighbours(data, p)
                # check neighbours length
                if len(neighbours_new) >= self.min_points:
                    neighbours = neighbours + neighbours_new
            i += 1


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import MinMaxScaler

    X, y = make_blobs(n_samples=30, centers=3, n_features=2)
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    model = DBSCAN()
    predictions = model.fit_predict(X)
    colors = ['r', 'g', 'b', 'c', 'k', 'y']
    for classification, x in zip(predictions, X):
        color = colors[classification]
        plt.scatter(x[0], x[1], color=color, s=150, linewidths=5, zorder=10)
    plt.show()
