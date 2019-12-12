import numpy as np
from collections import Counter
from scipy.spatial.distance import pdist, squareform


class SSClustering(object):
    def __init__(self, eps: float, min_points: int):
        self.eps = eps  # eps is the radius of the ball in which we look for neighbours of the point
        self.min_points = min_points  # min_points is the minimum number of points required to form a dense region
        self.n_clusters = None

    def fit(self, ldata, labels, udata):
        X = np.vstack((ldata, udata))
        distance = squareform(pdist(X))  # pairwise distance
        n = X.shape[0]

        is_visited = np.zeros(n, dtype="bool")
        clusters = []

        for i in range(n):
            if is_visited[i]:
                continue

            is_visited[i] = True

            neighbors = set(np.where(distance[i] < self.eps)[0])
            if len(neighbors) < self.min_points:  # density check
                continue

            current_cluster = [i]
            while len(neighbors) > 0:
                neighbor_idx = neighbors.pop()
                if is_visited[neighbor_idx]:
                    continue

                is_visited[neighbor_idx] = True

                current_neighbors = set(np.where(distance[neighbor_idx] < self.eps)[0])
                if len(current_neighbors) >= self.min_points:
                    current_cluster.append(neighbor_idx)
                    neighbors.update(current_neighbors)

            clusters.append(current_cluster)

        self.n_clusters = len(clusters)
        labeled_size = ldata.shape[0]
        new_labels = np.zeros(n, dtype="int") - 1
        new_labels[:labeled_size] = labels

        for cluster_indices in clusters:
            cluster_indices = np.asarray(cluster_indices)
            labeled_cluster_indices = cluster_indices[cluster_indices < labeled_size]

            cluster_labels = labels[labeled_cluster_indices]
            labels_counter = Counter(cluster_labels).most_common()

            if (
                len(labels_counter) == 0
                or len(labels_counter) > 1
                and labels_counter[0][1] == labels_counter[1][1]
            ):
                continue

            main_label = labels_counter[0][0]
            unlabeled_cluster_indices = cluster_indices[cluster_indices >= labeled_size]
            new_labels[unlabeled_cluster_indices] = main_label

        X_train = X[new_labels != -1]
        y_train = new_labels[new_labels != -1]

        return X_train, y_train
