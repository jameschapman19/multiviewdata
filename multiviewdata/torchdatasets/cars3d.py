import glob
import os

import PIL
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_and_extract_archive


class CarsDataset(Dataset):
    def __init__(self, root: str, download: bool = False, train: bool = True):
        """

        :param root: Root directory of dataset
        :param download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        :param train:
        """
        self.resources = [
            "http://www.scottreed.info/files/nips2015-analogy-data.tar.gz"
        ]
        self.root = root
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )
        if train:
            self.dataset = dict(
                view_1=torch.load(os.path.join(self.raw_folder, "view_1.pt")),
                view_2=torch.load(os.path.join(self.raw_folder, "view_2.pt")),
            )
        else:
            pass

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def __getitem__(self, index):
        return {
            "views": (self.dataset["view_1"][index], self.dataset["view_2"][index]),
            "index": index,
        }

    def __len__(self):
        return self.v1.shape[0]

    def _check_raw_exists(self) -> bool:
        return os.path.exists(
            os.path.join(self.raw_folder, "nips2015-analogy-data.tar.gz")
        )

    def _check_exists(self) -> bool:
        return os.path.exists(
            os.path.join(self.raw_folder, "view_1.pt")
        ) and os.path.exists(os.path.join(self.raw_folder, "view_2.pt"))

    def download(self) -> None:
        """Download the data if it doesn't exist in processed_folder already."""

        if not self._check_raw_exists():
            os.makedirs(self.raw_folder, exist_ok=True)
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context
            # download files
            for url in self.resources:
                filename = url.rpartition("/")[2]
                download_and_extract_archive(
                    url, download_root=self.raw_folder, filename=filename
                )
        if self._check_exists():
            return
        print("Processing...")
        view_1, view_2 = get_cars3d(os.path.join(self.raw_folder, "data/cars/"))
        with open(os.path.join(self.raw_folder, "view_1.pt"), "wb") as f:
            torch.save(view_1, f)
        with open(os.path.join(self.raw_folder, "view_2.pt"), "wb") as f:
            torch.save(view_2, f)
        print("Done!")


# Shuffle the private information
def sample(n1=183 * 2, n2=24):
    idx = []
    for i in range(n1):
        idx.append(np.random.permutation(n2) + i * n2)

    return np.hstack(idx)


# Get the Car3D dataset
def get_cars3d(filedir="./data/cars/"):
    tmp = []
    for f in glob.glob(filedir + "*mesh.mat"):
        a = sio.loadmat(f)
        tt = np.zeros((a["im"].shape[3], a["im"].shape[4], 64, 64, 3))
        for i in range(a["im"].shape[3]):
            for j in range(a["im"].shape[4]):
                pic = PIL.Image.fromarray(a["im"][:, :, :, i, j])
                pic.thumbnail((64, 64), PIL.Image.ANTIALIAS)
                tt[i, j, :, :, :] = np.array(pic) / 255.0

        b = torch.tensor(tt)
        c = b.permute(0, 1, 4, 2, 3)
        tmp.append(c)

    data = torch.stack(tmp, dim=0)

    imgs = data.numpy()
    imgs = np.transpose(imgs, (2, 0, 1, 3, 4, 5))

    # 4 elevations
    elv1 = imgs[0]
    elv2 = imgs[1]
    elv3 = imgs[2]
    elv4 = imgs[3]

    # Show samples
    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(np.transpose(elv1[0, 1, :, :, :], (1, 2, 0)))
    axarr[0].axis("off")
    axarr[1].imshow(np.transpose(elv2[0, 7, :, :, :], (1, 2, 0)))
    axarr[1].axis("off")
    axarr[2].imshow(np.transpose(elv3[0, 14, :, :, :], (1, 2, 0)))
    axarr[2].axis("off")
    axarr[3].imshow(np.transpose(elv4[0, 21, :, :, :], (1, 2, 0)))
    axarr[3].axis("off")

    fig.suptitle(
        "Samples of both views, left two for view1, right two for view2", fontsize=10
    )
    plt.show()

    # Get the two views in order
    view1 = []
    view2 = []

    for i in range(elv1.shape[0]):
        # Lower elevations
        view1.append(elv1[i, :, :, :, :])
        view1.append(elv2[i, :, :, :, :])
        # Higher elevations
        view2.append(elv3[i, :, :, :, :])
        view2.append(elv4[i, :, :, :, :])

    view1 = np.concatenate(view1, axis=0)
    view2 = np.concatenate(view2, axis=0)

    return view1, view2
