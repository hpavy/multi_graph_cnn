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


import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def plot_embedding_diagnostics(embedding_matrix, labels, title, save_path=None):
    """
    Diagnostique la qualité du clustering des embeddings W ou H.
    
    Args:
        embedding_matrix: La matrice W ou H (N_samples x Rank)
        labels: Les vrais labels (0 ou 1)
        title: Titre du plot (ex: "Diagnostics W (Users)")
    """
    # Convertir en numpy si c'est un tenseur PyTorch
    if hasattr(embedding_matrix, 'detach'):
        embedding_matrix = embedding_matrix.detach().cpu().numpy()
        
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- PLOT 1 : Matrice de Similarité (Cosinus) Triée ---
    # On calcule la similarité entre tous les vecteurs
    # Cela permet de voir si les vecteurs d'un même groupe sont proches
    sim_matrix = cosine_similarity(embedding_matrix) - np.eye(embedding_matrix.shape[0])
    
    # On trie les indices pour regrouper les labels identiques
    sort_idx = np.argsort(labels)
    sorted_sim = sim_matrix[sort_idx][:, sort_idx]
    
    im = axes[0].imshow(sorted_sim, cmap='viridis', interpolation='nearest')
    axes[0].set_title(f"{title} - Similarity Matrix (Sorted)")
    axes[0].set_xlabel("Sorted Index")
    axes[0].set_ylabel("Sorted Index")
    plt.colorbar(im, ax=axes[0])
    
    # --- PLOT 2 : Projection PCA 2D ---
    # On projette les 10 dimensions en 2D pour voir les nuages
    pca = PCA(n_components=2)
    embedding_2d = pca.fit_transform(embedding_matrix)
    
    scatter = axes[1].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                              c=labels, cmap='coolwarm', alpha=0.7, edgecolor='k')
    axes[1].set_title(f"{title} - PCA Projection (2D)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(*scatter.legend_elements(), title="Cluster")

    plt.suptitle(f"Analyse des Embeddings : {title}", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure sauvegardée : {save_path}")
    

