import numpy as np
import networkx as nx
from scipy.linalg import eigh

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