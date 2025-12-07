"""Import the data to feed the model"""

from scipy.io import loadmat
import scipy.sparse as sp
import numpy as np
import h5py

from multi_graph_cnn.utils import extract_nested_dict
from multi_graph_cnn.utils import get_logger

log = get_logger()

data_names = {
    "douban": "training_test_dataset.mat",
    "flixster": "training_test_dataset_10_NNs.mat",
    "movielens": "split_1.mat",
    "synthetic_netflix": "synthetic_netflix.mat",
    "yahoo_music": "training_test_dataset_10_NNs.mat",
}


def read_data(name_data):
    """
    Read the data and return an array
    Return a dict: 
        M: the rows are the users, the columns are the movies and the values are the rating
        O: same size as M. If we have a 0 we don't have a value. If we have a 1 we have a value
        O_training: same as O but without some values, to train the model
        O_test: same as O but without some values, to test the model
        We have O_train + O_test == O
        W_row: the adjency matrix of the user graph
        W_col: the adjency matrix of the movie graph
    """
    if name_data not in data_names:
        raise ValueError(
            f"Invalid data name: {name_data}. Valid options are: {list(data_names.keys())}"
        )
    file_path = "data/" + name_data + "/" + data_names[name_data]
    with h5py.File(file_path, "r") as f:
        data = extract_nested_dict(f, type_element=np.array)

    if name_data == "synthetic_netflix":
        M = np.asarray(data['M']).astype(np.float32).T
        O = np.asarray(data['O']).astype(np.float32).T
        O_training = np.asarray(data['Otraining']).astype(np.float32).T
        O_test = np.asarray(data['Otest']).astype(np.float32).T
        W_row = find_sparse_matrix(data["Wrow"])
        W_col = find_sparse_matrix(data["Wcol"])
        data = {
            "M": M,  # 150x200,
            "O": O,  # 150x200
            "O_training": O_training,  # 150x200
            "O_test": O_test,  # 150x200
            "W_row": W_row,  # 150x150
            "W_col": W_col  # 200x200
        }
    return data


def find_sparse_matrix(dict_values):
    """Find the sparse matrix from the representation we have"""
    data = np.asarray(dict_values['data'])
    ir = np.asarray(dict_values['ir'])
    jc = np.asarray(dict_values['jc'])
    return sp.csc_matrix((data, ir, jc)).astype(np.float32)


def split_data(data, config):
    """Split the data between train, test and val"""
    M = data["M"]
    O_test = data["O_test"]
    O_tot = data["O_training"]
    indices = np.where(O_tot)
    number_samples = len(indices[0])
    list_idx = [k for k in range(number_samples)]
    np.random.shuffle(list_idx)

    idx_tr_target = list_idx[:int(number_samples*config.rate_training)]
    idx_train = list_idx[int(number_samples*config.rate_training):]

    pos_tr_target_samples = (indices[0][idx_tr_target], indices[1][idx_tr_target])
    pos_tr_samples = (indices[0][idx_train], indices[1][idx_train])

    O_train_target = np.zeros_like(M)
    O_train = np.zeros_like(M)

    for k in range(len(pos_tr_target_samples[0])):
        O_train_target[pos_tr_target_samples[0][k], pos_tr_target_samples[1][k]] = 1

    for k in range(len(pos_tr_samples[0])):
        O_train[pos_tr_samples[0][k], pos_tr_samples[1][k]] = 1

    log.info(f"The number of element in O_train is: {int(np.sum(O_train).item())}")
    log.info(f"The number of element in O_train_target is: {int(np.sum (O_train_target).item())}")
    log.info(f"The number of element in O_test is: {int(np.sum(O_test).item())}")
    
    return O_train, O_train_target, O_test


def compute_the_laplacians(data):
    """Compute the laplacians"""
    W_row, W_col = data["W_row"], data["W_col"]
    L_row = sp.csgraph.laplacian(W_row, normed=True) #it is laplacian normalization but it does not mean its value are between -1 et 1
    L_col = sp.csgraph.laplacian(W_col, normed=True)
    return L_row, L_col
