import os

from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url


class XRMBDataset(Dataset):
    def __init__(
        self,
        root,
        train=True,
        download=False,
    ):
        """

        :param root: Root directory of dataset
        :param train:
        :param download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        """
        self.root = root
        self.resources = [
            ("http://ttic.edu/livescu/XRMB_data/full/XRMBf2KALDI_window7_single1.mat"),
            ("http://ttic.edu/livescu/XRMB_data/full/XRMBf2KALDI_window7_single2.mat"),
        ]
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )
        if train:
            view_1, view_2 = (
                loadmat("XRMBf2KALDI_window7_single1.mat")["X1"],
                loadmat("XRMBf2KALDI_window7_single2.mat")["X2"],
            )
        else:
            view_1, view_2 = (
                loadmat("XRMBf2KALDI_window7_single1.mat")["XTe1"],
                loadmat("XRMBf2KALDI_window7_single2.mat")["XTe2"],
            )
        self.dataset = dict(view_1=view_1, view_2=view_2)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def __len__(self):
        return len(self.view_1)

    def __getitem__(self, index):
        return {
            "views": (self.dataset["view_1"][index], self.dataset["view_2"][index]),
            "index": index,
        }

    def _check_exists(self) -> bool:
        return os.path.exists(
            os.path.join(self.raw_folder, "XRMBf2KALDI_window7_single1.mat")
        ) and os.path.exists(
            os.path.join(self.raw_folder, "XRMBf2KALDI_window7_single2.mat")
        )

    def download(self) -> None:
        """Download the data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context
        views = []
        for url in self.resources:
            filename = url.rpartition("/")[2]
            views.append(download_url(url, self.raw_folder, filename))
        print("Done!")
