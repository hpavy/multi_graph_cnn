"""Create two clusters and do stuff with them"""
import numpy as np
import random


def create_cluster(config):
    """Create two clusters"""
    user_tastes = np.random.choice([0, 1], size=config.nb_users)
    movie_kinds = np.random.choice([0, 1], size=config.nb_movies)
    return user_tastes, movie_kinds


import numpy as np

def create_exact_neighbours_graph(cluster, p_within=0.7, nb_neighbours=10, max_attempts=1000):
    """
    Crée un graphe où chaque nœud a exactement nb_neighbours voisins.
    La probabilité qu'un voisin soit dans le même cluster est p_within.
    """
    nb_elem = len(cluster)
    adjacency = np.zeros((nb_elem, nb_elem), dtype=int)
    degrees = np.zeros(nb_elem, dtype=int)

    attempts = 0
    while np.any(degrees < nb_neighbours) and attempts < max_attempts:
        i = np.random.choice(np.where(degrees < nb_neighbours)[0])
        # Déterminer si le voisin sera dans le même cluster ou non
        if np.random.rand() < p_within:
            candidates = np.where((cluster == cluster[i]) & (degrees < nb_neighbours) & (adjacency[i]==0) & (np.arange(nb_elem)!=i))[0]
        else:
            candidates = np.where((cluster != cluster[i]) & (degrees < nb_neighbours) & (adjacency[i]==0))[0]

        if len(candidates) == 0:
            attempts += 1
            continue

        j = np.random.choice(candidates)

        # Crée le lien symétrique
        adjacency[i, j] = 1
        adjacency[j, i] = 1
        degrees[i] += 1
        degrees[j] += 1
        attempts = 0  

    if np.any(degrees < nb_neighbours):
        print("Attention : certains nœuds n'ont pas pu atteindre le nombre exact de voisins.")

    return adjacency

def generate_data(user_tastes, movie_kinds, config):
    """Create the data based on the clusters and the stochasticity"""
    ranking_matrix = np.zeros((user_tastes.shape[0], movie_kinds.shape[0]))
    for user_id in range(len(user_tastes)):
        watched_movies = np.random.choice(len(movie_kinds), config.nb_ranked)
        for movie_id in watched_movies:
            if user_tastes[user_id] == movie_kinds[movie_id]:
                ranking = config.mean_good_ranked + random.uniform(
                    -config.variance_ranking, config.variance_ranking
                    )
            else:
                ranking = config.mean_bad_ranked + random.uniform(
                    -config.variance_ranking, config.variance_ranking
                    )
            ranking_matrix[user_id, movie_id] = ranking
    return ranking_matrix

def create_stochastic_graph(cluster, p_within, nb_neigbours):
    """
    Create a graph where p_within is the probability to be linked to an element in the same cluster
    And p_between is the probability to be linked to an element in an other cluster
    """
    nb_elem = len(cluster)
    adjacency = np.zeros((nb_elem, nb_elem))

    for i in range(nb_elem):
        probs = np.where(cluster == cluster[i], p_within, 1 - p_within)
        probs = probs / probs.sum()
        neigbours = np.random.choice(len(probs), size=nb_neigbours, replace=False, p=probs)
        neigbours = neigbours[neigbours != i]  # In case we take the node
        adjacency[i][neigbours] = 1.
    return adjacency + adjacency.T  # Maybe we can find a better way to do that ()


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.linalg as la

def plot_graph_diagnostics(adj_matrix, labels,save_directory):
    """
    Plots diagnostics to check if a graph structure matches the labels.
    
    Args:
        adj_matrix: The adjacency matrix (graph_movies)
        labels: The ground truth cluster labels (movie_kinds)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # --- Plot 1: Sorted Adjacency Matrix ---
    # Sort the matrix indices based on the labels
    sort_idx = np.argsort(labels)
    sorted_adj = adj_matrix[sort_idx][:, sort_idx]
    
    axes[0].imshow(sorted_adj, cmap='Greys', interpolation='nearest', aspect='auto')
    axes[0].set_title("Adjacency Matrix (Sorted by Label)")
    axes[0].set_xlabel("Nodes")
    axes[0].set_ylabel("Nodes")

    # --- Plot 2: Sorted Fiedler Vector ---
    # Compute Normalized Laplacian
    degrees = np.sum(adj_matrix, axis=1)
    # Avoid div by zero
    degrees[degrees == 0] = 1 
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L_norm = np.eye(len(degrees)) - d_inv_sqrt @ adj_matrix @ d_inv_sqrt
    
    # Get 2nd smallest eigenvector
    evals, evecs = la.eigh(L_norm, subset_by_index=[1, 1])
    fiedler_vec = evecs.flatten()
    
    # Sort the vector values
    axes[1].plot(np.sort(fiedler_vec), marker='o', markersize=2, linestyle='-')
    axes[1].set_title("Sorted Fiedler Vector")
    axes[1].set_xlabel("Node Index")
    axes[1].set_ylabel("Eigenvector Value")
    axes[1].grid(True, alpha=0.3)

    # --- Plot 3: Network Visualization ---
    # Create NetworkX graph
    G = nx.from_numpy_array(adj_matrix)
    
    # Use spring layout (force-directed)
    # We only compute positions once
    pos = nx.spring_layout(G, iterations=50, seed=42)
    
    # Draw nodes colored by label
    nx.draw_networkx_nodes(G, pos, ax=axes[2], node_size=30, 
                           node_color=labels, cmap='coolwarm', alpha=0.8)
    # Optional: Draw edges (can be messy if dense)
    # nx.draw_networkx_edges(G, pos, ax=axes[2], alpha=0.05)
    
    axes[2].set_title("Force-Directed Layout")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_directory)





# --- USAGE ---
# Assuming you have these variables from your main.py:
# plot_graph_diagnostics(graph_movies, movie_kinds)