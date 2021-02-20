import numpy as np
import random


class KMeans:
    def __init__(self, n_clusters=2, tol=0.0001, max_iter=300):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
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

    def predict(self, data):
        classifications = []
        for row in data:
            distances = [np.linalg.norm(row-self.centroids[centroid])
                         for centroid in self.centroids]
            classification = distances.index(min(distances))
            classifications.append(classification)
        return classifications
