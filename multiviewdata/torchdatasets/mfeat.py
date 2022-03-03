import os

import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_and_extract_archive


class MFeatDataset(Dataset):
    def __init__(
        self,
        root: str,
        feats: list = None,
        partials: list = None,
        download: bool = False,
    ):
        """

        :param root: Root directory of dataset
        :param feats: Which features to use from ["fac", "fou", "kar", "mor", "pix", "zer"]
        :param partials: Which features to use as partials from ["fac", "fou", "kar", "mor", "pix", "zer"]
        :param download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        """
        self.resources = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat.tar"
        ]
        self.root = root
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )
        if feats is None:
            self.feats = ["fac", "fou", "kar", "mor", "pix", "zer"]
        if partials is None:
            self.partials = None
        self.dataset = dict(
            fac=np.genfromtxt(os.path.join(self.raw_folder, "mfeat/mfeat-fac")),
            fou=np.genfromtxt(os.path.join(self.raw_folder, "mfeat/mfeat-fou")),
            kar=np.genfromtxt(os.path.join(self.raw_folder, "mfeat/mfeat-kar")),
            mor=np.genfromtxt(os.path.join(self.raw_folder, "mfeat/mfeat-mor")),
            pix=np.genfromtxt(os.path.join(self.raw_folder, "mfeat/mfeat-pix")),
            zer=np.genfromtxt(os.path.join(self.raw_folder, "mfeat/mfeat-zer")),
        )

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def __getitem__(self, index):
        batch = {"index": index}
        batch["views"] = [self.dataset[feat][index].astype(np.float32) for feat in self.feats]
        if self.partials is not None:
            batch["partials"] = [self.dataset[partial][index].astype(np.float32) for partial in self.partials]
        return batch

    def __len__(self):
        return len(self.dataset["fac"])

    def _check_raw_exists(self) -> bool:
        return os.path.exists(os.path.join(self.raw_folder, "mfeat.tar"))

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.raw_folder, "mfeat"))

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
