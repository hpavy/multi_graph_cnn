from clustering import create_cluster
from multi_graph_cnn.utils import load_config
import numpy as np

def save_arrays(user_tastes, movie_kinds, filename="clustering_data.npz"):
    """Save user_tastes and movie_kinds arrays to a .npz file."""
    np.savez(filename, user_tastes=user_tastes, movie_kinds=movie_kinds)

def load_arrays(filename="clustering_data.npz"):
    """Load user_tastes and movie_kinds arrays from a .npz file."""
    data = np.load(filename)
    return data['user_tastes'], data['movie_kinds']


if __name__ == "__main__":
    np.random.seed(42)

    config = load_config("find_imapct_graph/impact_graph.yaml")
    user_tastes = np.random.choice([0, 1], size=config.nb_users)
    movie_kinds = np.random.choice([0, 1], size=config.nb_movies)
    save_arrays(user_tastes, movie_kinds, "find_impact_graph/data.npz")


