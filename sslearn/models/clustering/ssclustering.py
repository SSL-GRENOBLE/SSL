import numpy as np
from collections import Counter
from scipy.spatial.distance import pdist, squareform


class SSClustering(object):
    def __init__(
        self, eps: float, min_points: int, use_border_points: bool = False
    ) -> None:
        """
        Arguments:
            eps: Radius of the L2-ball in which we look for neighbours of the point.
            min_points: Minimum number of points required to form a dense region.
        """

        self.eps = eps
        self.min_points = min_points
        self.use_border_points = use_border_points
        self.n_clusters = None

    def transform(self, ldata, labels, udata):
        """
        Arguments:
            ldata: Labelled data.
            labels: Labels for labelled data.
            udata: Unlabelled data.
        """

        X = np.vstack((ldata, udata))
        distance = squareform(pdist(X))  # pairwise distance
        total_size = X.shape[0]

        # -1 - not visited, 0 - noise, >0 - cluster number
        point_type = np.zeros(total_size, dtype="int") - 1
        is_border_point = np.zeros(total_size, dtype="bool")
        n_clusters = 0

        for point_idx in range(total_size):
            if point_type[point_idx] != -1:
                continue

            neighbors = set(np.where(distance[point_idx] < self.eps)[0])
            if len(neighbors) < self.min_points:  # density check
                point_type[point_idx] = 0
                continue

            n_clusters += 1
            point_type[point_idx] = n_clusters

            while len(neighbors) > 0:
                neighbor_idx = neighbors.pop()

                if point_type[neighbor_idx] == 0:
                    point_type[neighbor_idx] = n_clusters
                    is_border_point[neighbor_idx] = True

                if point_type[neighbor_idx] != -1:
                    continue

                point_type[neighbor_idx] = n_clusters

                current_neighbors = set(np.where(distance[neighbor_idx] < self.eps)[0])
                if len(current_neighbors) >= self.min_points:
                    neighbors.update(current_neighbors)
                else:
                    is_border_point[neighbor_idx] = True

        self.n_clusters = n_clusters
        labeled_size = ldata.shape[0]
        labeled_mask = np.arange(total_size) < labeled_size  # bool np.array
        new_labels = np.zeros(total_size, dtype="int") - 1
        new_labels[labeled_mask] = labels

        for cluster_idx in range(1, self.n_clusters + 1):
            cluster_mask = point_type == cluster_idx
            if not self.use_border_points:
                cluster_mask &= ~is_border_point

            cluster_labeled_indices = (cluster_mask & labeled_mask)[:labeled_size]
            cluster_unlabeled_indices = cluster_mask & ~labeled_mask

            cluster_labels = labels[cluster_labeled_indices]
            labels_counter = Counter(cluster_labels).most_common()

            if (
                len(labels_counter) == 0
                or len(labels_counter) > 1
                and labels_counter[0][1] == labels_counter[1][1]
            ):
                continue

            main_label = labels_counter[0][0]
            new_labels[cluster_unlabeled_indices] = main_label

        result_mask = new_labels != -1
        return X[result_mask], new_labels[result_mask]
