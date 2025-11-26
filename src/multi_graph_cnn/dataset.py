"""Create a pytorch Dataset in order to train the model"""

from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, data, config):
        self.data = data

    def __len__(self):
        pass # TODO

    def __getitem__(self, idx):
        pass
