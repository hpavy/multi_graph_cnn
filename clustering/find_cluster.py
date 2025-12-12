import numpy as np
import networkx as nx
from scipy.linalg import eigh
import scipy.linalg as la
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def find_cluster_from_graph(A):
    """Find the best cluster possible from a graph using Normalized Laplacian"""

    degrees = np.sum(A, axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    
    D_inv_sqrt = np.diag(d_inv_sqrt)
    I = np.eye(A.shape[0])
    L_norm = I - D_inv_sqrt @ A @ D_inv_sqrt
    eigenvalues, eigenvectors = eigh(L_norm, subset_by_index=[0, 1])

    fiedler_vector = eigenvectors[:, 1]

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

    if hasattr(X, 'detach'):
        X = X.detach().cpu().numpy()
    clustering = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
    labels = clustering.fit_predict(X)
    return labels

def clustering_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc1 = (y_true == y_pred).mean()

    return max(acc1, 1-acc1)


def plot_embedding_diagnostics(embedding_matrix, labels, title, save_path=None):
    """
    Diagnostique la qualité du clustering des embeddings W ou H.
    
    Args:
        embedding_matrix: La matrice W ou H (N_samples x Rank)
        labels: Les vrais labels (0 ou 1)
        title: Titre du plot (ex: "Diagnostics W (Users)")
    """
    if hasattr(embedding_matrix, 'detach'):
        embedding_matrix = embedding_matrix.detach().cpu().numpy()
        
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- PLOT 1 : Matrice de Similarité (Cosinus) Triée ---
    sim_matrix = cosine_similarity(embedding_matrix) - np.eye(embedding_matrix.shape[0])
    sort_idx = np.argsort(labels)
    sorted_sim = sim_matrix[sort_idx][:, sort_idx]
    
    im = axes[0].imshow(sorted_sim, cmap='viridis', interpolation='nearest')
    axes[0].set_title(f"{title} - Similarity Matrix (Sorted)")
    axes[0].set_xlabel("Sorted Index")
    axes[0].set_ylabel("Sorted Index")
    plt.colorbar(im, ax=axes[0])
    
    # --- PLOT 2 : Projection PCA 2D ---
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
    


def plot_graph_diagnostics(adj_matrix, labels,save_directory):
    """
    Plots diagnostics to check if a graph structure matches the labels.
    
    Args:
        adj_matrix: The adjacency matrix
        labels: The ground truth cluster labels
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Sorted Adjacency Matrix ---
    sort_idx = np.argsort(labels)
    sorted_adj = adj_matrix[sort_idx][:, sort_idx]
    
    axes[0].imshow(sorted_adj, cmap='Greys', interpolation='nearest', aspect='auto')
    axes[0].set_title("Adjacency Matrix (Sorted by Label)")
    axes[0].set_xlabel("Nodes")
    axes[0].set_ylabel("Nodes")

    # --- Plot 2: Sorted Fiedler Vector ---
    degrees = np.sum(adj_matrix, axis=1)
    degrees[degrees == 0] = 1 
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L_norm = np.eye(len(degrees)) - d_inv_sqrt @ adj_matrix @ d_inv_sqrt
    evals, evecs = la.eigh(L_norm, subset_by_index=[1, 1])
    fiedler_vec = evecs.flatten()
    
    axes[1].plot(np.sort(fiedler_vec), marker='o', markersize=2, linestyle='-')
    axes[1].set_title("Sorted Fiedler Vector")
    axes[1].set_xlabel("Node Index")
    axes[1].set_ylabel("Eigenvector Value")
    axes[1].grid(True, alpha=0.3)

    # --- Plot 3: Network Visualization ---
    G = nx.from_numpy_array(adj_matrix)
    pos = nx.spring_layout(G, iterations=50, seed=42)
    nx.draw_networkx_nodes(G, pos, ax=axes[2], node_size=30, 
                           node_color=labels, cmap='coolwarm', alpha=0.8)
    
    axes[2].set_title("Force-Directed Layout")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_directory)