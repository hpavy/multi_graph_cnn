import numpy as np
import networkx as nx
from scipy.linalg import eigh
from sklearn.cluster import KMeans, SpectralClustering

def find_cluster_from_graph(A):
    """Find the best cluster possible from a graph using Normalized Laplacian"""
    # 1. Ensure the matrix is symmetric (just in case) and fix weights
    # Convert weights of 2 (mutual connection) to 1 if you want binary, 
    # or keep them if you want to reward mutual links. 
    # Here we stick to the input A but ensure symmetry logic holds.
    
    # 2. Compute Degree Matrix
    degrees = np.sum(A, axis=1)
    # 3. Compute Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
    # Handle division by zero for isolated nodes
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    # Identity matrix
    I = np.eye(A.shape[0])
    
    # Normalized Laplacian
    L_norm = I - D_inv_sqrt @ A @ D_inv_sqrt

    # 4. Compute eigenvalues and eigenvectors
    # We want the eigenvector for the 2nd *smallest* eigenvalue
    eigenvalues, eigenvectors = eigh(L_norm, subset_by_index=[0, 1])

    # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
    fiedler_vector = eigenvectors[:, 1]
    
    # Handle the case where sign is 0 (rare)
    clusters = np.where(fiedler_vector > 0, 1, 0)
    
    return clusters

def cluster_linear_embedding(X):
    # KMeans works on numpy arrays
    if hasattr(X, 'detach'):
        X = X.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels

def cluster_Spectral_embedding(X):
    # Spectral Clustering on embeddings
    if hasattr(X, 'detach'):
        X = X.detach().cpu().numpy()
    
    # NOTE: For dense embeddings (like W and H from SVD), KMeans is often 
    # more stable than SpectralClustering with nearest_neighbors affinity,
    # unless the manifold is very non-linear. 
    # If this remains low, try switching to KMeans here too.
    clustering = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
    labels = clustering.fit_predict(X)
    return labels

def clustering_accuracy(y_true, y_pred):
    """
    y_true : initial labels (size n)
    y_pred : predicted labels (size n)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc1 = (y_true == y_pred).mean()

    return max(acc1, 1-acc1)