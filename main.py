import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


class KMeans:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    def euclidean_distance(self, data_point, centroids):
        '''calculate the distance between a data point and all the centroids'''
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

    def fit(self, X, max_iterations=200):
        # keep the centroids in the boundaries of the existing data
        # keep the centrois within the min and max of this respective dimension
        # initializing the centroinds randomly
        self.centroids = np.random.uniform(
            np.amin(X, axis=0),
            np.amax(X, axis=0),
            size=(self.k, X.shape[1])
        ) # for each dimension it is considered the range from min to max and generate k centroids

        print(f'centroids: {self.centroids}')

        for _ in range(max_iterations):
            y = [] # clusters' labels

            for data_point in X:
                # assign each data point to the closest centroid
                distances = self.euclidean_distance(data_point, self.centroids)
                cluster_index = np.argmin(distances)
                y.append(cluster_index) # storing the index of the data point cluster (nearest centroid to that data point)

            y = np.array(y)

            cluster_indices = [] # list of lists, it will store the list of indices of data points that belongs the that cluster

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i)) # storing the list of indices

            clusters_centers = []

            # reposition the centroid to the middle of all the points that are assigned to that cluster
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0: # empty cluster
                    clusters_centers.append(self.centroids[i])
                else:
                    clusters_centers.append(np.mean(X[indices], axis=0)[0]) # store the mean position

            if np.max(self.centroids - np.array(clusters_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(clusters_centers)
        
        return y


# rand_points = np.random.randint(0, 100, (100, 2))
data = make_blobs()
rand_points = data[0]
kmeans = KMeans(3)
labels = kmeans.fit(rand_points)
print(f'Original points: {rand_points}')
print(f'Clusters labels: {labels}')
print(f'Original labels: {data[1]}')
print(f'Rand score: {adjusted_rand_score(data[1], labels)}')

plt.scatter(rand_points[:, 0], rand_points[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)),
    marker="*", s=200)
plt.show()
