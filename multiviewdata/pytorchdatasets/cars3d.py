import torch
from torch.utils.data import Dataset

from multiviewdata.utils.cars3d import get_cars3d


class Cars_Dataset(Dataset):
    def __init__(self):
        v1, v2 = get_cars3d
        self.v1 = torch.tensor(v1).float()
        self.v2 = torch.tensor(v2).float()
        self.data_len = v1.shape[0]

    def __getitem__(self, index):
        return {"views": (self.v1[index], self.v2[index]), "index": index}

    def __len__(self):
        return self.data_len
