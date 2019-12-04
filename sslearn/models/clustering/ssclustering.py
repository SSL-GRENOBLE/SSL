import numpy as np
from scipy.spatial.distance import pdist, squareform


class SSClustering(object):
    def __init__(self, eps: float, min_points: int):
        """
        """
        self.eps = eps  # eps is the radius of the ball in which we look for neighbours of the point
        self.min_points = min_points  # min_points is the minimum number of points required to form a dense region
        self.n_clusters = None

    def fit(self, ldata, labels, udata):
        """
        """
        # DF is a m x n matrix, where m is number of data for training and n is the number of features, it it x_u_train
        DF, x_l_train, y_l_train = udata, ldata, labels
        # Initialization
        DF_merged = np.concatenate((DF, x_l_train))
        distance = squareform(pdist(DF_merged))
        # The square matrix with distances from each point to each point
        # print(np.mean(distance))
        m = DF_merged.shape[
            0
        ]  # the amount of data to be divided into clusters (number of rows in DF)
        visit = np.zeros(
            m, "int"
        )  # the array in which to store whether the point was visited (1) or not (0)
        # initially all points are not visited, so the value  = 0
        point_type = np.zeros(
            m
        )  # the array: if 0 then point is border of cluster, if 1 then point is the core point of the cluster
        # if 2 then point is noise
        cluster = []  # the array of points of certain cluster
        clusters = []  # The list of clusters
        point_cluster_index = np.zeros(
            m
        )  # the array containing the cluster number of each point
        cluster_index = 0  # cluster counter
        Current_neighbors = []
        for i in range(m):  # We want to view all points in DF
            if visit[i] == 0:  # If the point i has not been visited yet
                visit[i] = 1  # label the point as visited
                Current_neighbors = np.where(distance[i] < self.eps)[
                    0
                ]  # Find neighbors
                if len(Current_neighbors) < self.min_points:  # density check
                    point_type[i] = 2  # label the point as noise
                else:
                    cluster_index += 1  # a new cluster has formed
                    cluster.append(i)  # add i to new cluster
                    point_cluster_index[i] = cluster_index  # change the point label
                    Current_neighbors_list = []
                    for v in Current_neighbors:
                        Current_neighbors_list.append(v.tolist())
                    neighbors = []
                    for b in Current_neighbors_list:
                        if visit[b] == 0:
                            visit[b] = 1
                            neighbors = np.where(distance[b] < self.eps)[0]
                            if len(neighbors) >= self.min_points:
                                neighbors_list = []
                                for p in neighbors:
                                    neighbors_list.append(p.tolist())
                                for q in range(len(neighbors_list)):
                                    if neighbors_list[q] not in Current_neighbors_list:
                                        Current_neighbors_list.append(neighbors_list[q])
                        if point_cluster_index[b] == 0:
                            cluster.append(b)
                            point_cluster_index[b] = cluster_index

                    cluster.append(Current_neighbors_list[:])
                    clusters.append(cluster[:])
        # print(cluster_index)
        self.cluster_index = cluster_index
        self.point_cluster_index = point_cluster_index

        m_u = DF.shape[0]
        m_l = x_l_train.shape[0]
        res = np.zeros(m_l)
        res = point_cluster_index[m_u:m]
        chan = np.zeros(cluster_index + 1)
        for i in range(0, cluster_index + 1):
            if i == 0:
                chan[i] = -1
            elif i in res:
                k = 0
                for j in range(len(res)):
                    if int(res[j] == i):
                        if int(y_l_train[j]) == 0:
                            k = k - 1
                        elif int(y_l_train[j]) == 1:
                            k = k + 1
                if k < 0:
                    chan[i] = 0
                elif k > 0:
                    chan[i] = 1
                else:
                    chan[i] = -1
            else:
                chan[i] = -1
        for i in range(0, cluster_index + 1):
            point_cluster_index[point_cluster_index == i] = chan[i]
        x_new = []
        for i in range(0, m):
            if (point_cluster_index[i] == 0) or (point_cluster_index[i] == 1):
                x_new.append(i)
        self.point_cluster_index = point_cluster_index
        self.x_new = x_new

        X_train_1 = []
        Y_train_1 = []
        for i in range(len(x_new)):
            X_train_1.append(DF_merged[x_new[i]])
            Y_train_1.append(point_cluster_index[x_new[i]])
        X_train_1 = np.array(X_train_1)
        Y_train_1 = np.array(Y_train_1)

        return (X_train_1, Y_train_1)
