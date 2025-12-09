import numpy as np
import networkx as nx
from scipy.linalg import eigh
from sklearn.cluster import KMeans, SpectralClustering

def find_cluster_from_graph(A):
    """Find the best cluster possible from a graph"""
    # Compute Laplacian
    D = np.diag(np.sum(A, axis=1))
    L = D - A

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(L, subset_by_index=[0, 1])

    # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
    fiedler_vector = eigenvectors[:, 1]

    # Assign clusters based on the sign of the Fiedler vector
    clusters = np.sign(fiedler_vector)
    return clusters 

def cluster_linear_embedding(X):
    # KMeans de scikit-learn fonctionne sur des numpy arrays
    X = X.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X)
    return(labels)

def cluster_Spectral_embedding(X):
    # KMeans de scikit-learn fonctionne sur des numpy arrays
    X = X.detach().cpu().numpy()
    clustering = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
    labels = clustering.fit_predict(X)
    return(labels)

def clustering_accuracy(y_true, y_pred):
    """
    y_true : labels initiaux (taille n)
    y_pred : labels prédits (taille n)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc1 = (y_true == y_pred).mean()

    acc2 = (y_true == (1 - y_pred)).mean()

    return max(acc1, acc2)