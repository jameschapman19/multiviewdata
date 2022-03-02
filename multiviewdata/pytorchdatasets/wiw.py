import numpy as np
from torch.utils.data import Dataset


class WIW_Dataset(Dataset):
    def __init__(self, feats=None, lng_dict=None, split="train"):
        # Dataset parameters
        if lng_dict is None:
            lng_dict = {
                "eng": "english",
                "ger": "german",
                "it": "italian",
                "ru": "russian",
            }
        if feats is None:
            feats = ["eng", "ru", "vis"]
        self.feats = feats
        self.lng_dict = lng_dict
        self.folder = self.feats[0] + "_" + self.feats[1]
        self.dname = "wiw_" + self.folder + "_img_pca_300_wv_splitted"  # Dataset Name
        self.dataset = h5py.File(
            "C:/Users/chapm/PycharmProjects/PartialBarlowTwins/data/wiw_data/"
            + self.folder
            + "/"
            + self.dname
            + ".h5",
            "r",
        )[split]
        self.comment = ""

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        views = [
            self.dataset["%06d" % idx][feat + "_feats"][()].astype(np.float32)
            for feat in self.feats
        ]
        return {"views": tuple(views[:-1]), "partials": views[-1]}
