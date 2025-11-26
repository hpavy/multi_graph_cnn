"""Import the data to feed the model"""
from scipy.io import loadmat
import numpy as np
import h5py


def read_data(name_data):
    """Read the data and return an array"""
    data_names = {
        "douban": "training_test_dataset.mat",
        "flixster": "training_test_dataset_10_NNs.mat",
        "movielens": "split_1.mat",
        "synthetic_netflix": "synthetic_netflix.mat",
        "yahoo_music": "training_test_dataset_10_NNs.mat"
    }
    if name_data not in data_names:
        raise ValueError(
            f"Invalid data name: {name_data}. Valid options are: {list(data_names.keys())}"
            )
    file_path = "data/" + name_data + "/" + data_names[name_data]
    with h5py.File(file_path, 'r') as f:
        keys = f.keys()
        data = {key: np.array(f[key]) for key in keys}  # Maybe we need to put a transpose here...
    return data

def split_data(data, config):
    """Split the data between train, test and val"""
    pass # TODO

if __name__ == "__main__":
    data = read_data("yahoo_music")
    print("oh")