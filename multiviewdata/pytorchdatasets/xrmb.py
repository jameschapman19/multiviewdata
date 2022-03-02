from torch.utils.data import Dataset

from multiviewdata.utils.xrmb import load_xrmb


class XRMB_Dataset(Dataset):
    def __init__(
        self,
        mode="train",
    ):
        view_1, view_2 = load_xrmb()
        if mode == "train":
            self.view_1 = view_1["X1"]
            self.view_2 = view_2["X2"]
        elif mode == "val":
            self.view_1 = view_1["XV1"]
            self.view_2 = view_2["XV2"]
        elif mode == "test":
            self.view_1 = view_1["XTe1"]
            self.view_2 = view_2["XTe2"]

    def __len__(self):
        return len(self.view_1)

    def __getitem__(self, idx):
        return {"views": (self.view_1[idx], self.view_2[idx])}
