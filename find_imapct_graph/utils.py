import numpy as np
import scipy as sp


def split_ranking(ranking, config):
    """Split the ranking between train, target, test"""
    O_tot = (ranking != 0).astype(int)
    O_train_target, O_test = split_O(O_tot, (1-config.rate_test))
    O_train, O_target = split_O(O_train_target, config.rate_training)
    return O_train, O_target, O_test


def split_O(O_tot, rate_split):
    """Split the O_tot matrix between two matrix randomly"""
    indices = np.where(O_tot)
    number_samples = len(indices[0])
    list_idx = [k for k in range(number_samples)]
    np.random.shuffle(list_idx)

    idx_tr_target = list_idx[:int(number_samples*rate_split)]
    idx_train = list_idx[int(number_samples*rate_split):]

    pos_tr_target_samples = (indices[0][idx_tr_target], indices[1][idx_tr_target])
    pos_tr_samples = (indices[0][idx_train], indices[1][idx_train])

    O_1 = np.zeros_like(O_tot)
    O_2 = np.zeros_like(O_tot)

    for k in range(len(pos_tr_target_samples[0])):
        O_1[pos_tr_target_samples[0][k], pos_tr_target_samples[1][k]] = 1

    for k in range(len(pos_tr_samples[0])):
        O_2[pos_tr_samples[0][k], pos_tr_samples[1][k]] = 1

    return O_1, O_2


def compute_the_laplacians(graph):
    """Compute the laplacian from a graph"""
    return sp.sparse.csgraph.laplacian(graph, normed=True)
