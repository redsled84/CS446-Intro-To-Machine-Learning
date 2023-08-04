#Author: Lucas Black
#Date: 10.18.22

import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

def K_Means(X, K, mu=None):
    # get K random cluster centers
    cluster_centers = X[np.random.choice(X.shape[0],
        size=K,
        replace=False)]
    # ensure clusters are contained within an array
    if not ('array' in str(type(cluster_centers[0]))):
        cluster_centers = np.array([np.array([center]) for center in cluster_centers])

    # initialize cluster centers to mu
    if not (mu is None):
        # handle when given cluster centers don't match K
        if len(mu) > K:
            mu = mu[:K]
        cluster_centers = mu

    # keep track of previous cluster centers for convergence
    prev_cluster_centers = np.ones(K)

    # repeat until convergence
    while not np.array_equal(prev_cluster_centers, cluster_centers):
        prev_cluster_centers = np.copy(cluster_centers)

        # list to store K clusters
        K_clusters = [[] for k in range(0, K)]

        # iterate over every data point
        for indx, value in enumerate(X):
            # record the distance from each sample point
            # to each cluster center
            dist = []
            for indx,center in enumerate(cluster_centers):
                type_value = str(type(value))

                # handle X that has one column
                # euclidean() only handles 1-D arrays
                _value = value
                if not ('array' in type_value):
                    _value = [value]

                dist.append(euclidean(center, _value))

            # convert to numpy array to use argmin()
            dist = np.array(dist)

            # append data point to closest cluster
            K_clusters[dist.argmin()].append(value)

        # recompute cluster centers
        for k, cluster in enumerate(K_clusters):
            cluster_centers[k] = np.array(sum(cluster)) / len(cluster)

    return cluster_centers

def writeup_plot(X, K, mu=None):
    colors = ["#0ff0ff", "#ff0ff0", "#fae000"]
    cluster_centers = K_Means(X, K, mu)

    # plot samples
    for x in X:
        dist = [[] for k in range(K)]

        for indx, center in enumerate(cluster_centers):
            dist[indx] = euclidean(center, x)

        dist = np.array(dist)

        plt.scatter(x[0], x[1], c=colors[dist.argmin()])

    # plot cluster centers
    for indx, center in enumerate(cluster_centers):
        plt.scatter(center[0], center[1], marker='+')
        plt.annotate("Cluster " + str(indx + 1), center)

    plt.show()
