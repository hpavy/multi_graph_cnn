"""Import the data to feed the model"""

from scipy.io import loadmat
import scipy.sparse as sp
import numpy as np
import h5py

from multi_graph_cnn.utils import extract_nested_dict

data_names = {
    "douban": "training_test_dataset.mat",
    "flixster": "training_test_dataset_10_NNs.mat",
    "movielens": "split_1.mat",
    "synthetic_netflix": "synthetic_netflix.mat",
    "yahoo_music": "training_test_dataset_10_NNs.mat",
}


def read_data(name_data):
    """Read the data and return an array"""
    if name_data not in data_names:
        raise ValueError(
            f"Invalid data name: {name_data}. Valid options are: {list(data_names.keys())}"
        )
    file_path = "data/" + name_data + "/" + data_names[name_data]
    with h5py.File(file_path, "r") as f:
        data = extract_nested_dict(f, type_element=np.array)
    if name_data == "synthetic_netflix":
        M = np.asarray(data['M']).astype(np.float32).T
        O_training = np.asarray(data['Otraining']).astype(np.float32).T
        O_test = np.asarray(data['Otest']).astype(np.float32).T
        W_row = find_sparse_matrix(data["Wrow"])
        W_col = find_sparse_matrix(data["Wcol"])
        data = {
            "M": M,  # 150x200
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
    pass  # TODO


if __name__ == "__main__":
    data = read_data("synthetic_netflix")
    print("oh")
