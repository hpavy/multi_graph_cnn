"""Create two clusters and do stuff with them"""
import numpy as np
import random



def create_cluster(config):
    """Create two clusters"""
    user_tastes = np.random.choice([0, 1], size=config.nb_users)
    movie_kinds = np.random.choice([0, 1], size=config.nb_movies)
    return user_tastes, movie_kinds

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
