import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


def determine_eps(data, k=4):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data)
    distances, indices = neigh.kneighbors(data)

    sorted_distances = np.sort(distances[:, -1])
    kneedle = KneeLocator(range(len(sorted_distances)), sorted_distances, curve="convex", direction="increasing")
    optimal_eps = sorted_distances[kneedle.knee]
    return optimal_eps

def predict(db, x):
    dists = np.sqrt(np.sum((db.components_ - x)**2, axis=1))
    i = np.argmin(dists)
    return db.labels_[db.core_sample_indices_[i]] if dists[i] < db.eps else -1

def create_dbscan(latent_points, latent_dim):
    latent_points = np.array(latent_points)
    min_samples = 2 * latent_dim  # Minimum number of points to form a cluster
    eps = determine_eps(latent_points, min_samples)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(latent_points)
    return dbscan

def dbscan_classifier(db, point):
    label = predict(db, point)
    if label < 0:
        return 1
    else:
        return 0

def show_cluster(latent_points, points_mapped, latent_dim):

    latent_points = np.array(latent_points)
    eps = determine_eps(latent_points, 2 * latent_dim)
    points_mapped = np.array(points_mapped)

    plt.scatter(latent_points[:, 0], latent_points[:, 1], s=20)
    plt.title("Original Data")
    plt.show()

    min_samples = 2 * latent_dim
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(latent_points)
    # labels = dbscan.labels_

    clusters = dbscan.fit_predict(points_mapped)

    plt.scatter(points_mapped[:, 0], points_mapped[:, 1], c=clusters, cmap="viridis", s=20)
    plt.title("DBSCAN Clustering")
    plt.colorbar(label="Cluster Label")
    plt.show()
