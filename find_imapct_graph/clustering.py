"""Create two clusters and do stuff with them"""
import numpy as np
import random


def create_cluster(config):
    """Create two clusters"""
    user_tastes = np.random.choice([0, 1], size=config.nb_users)
    movie_kinds = np.random.choice([0, 1], size=config.nb_movies)
    return user_tastes, movie_kinds


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
