import os

from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_and_extract_archive


class MIRFlickr(Dataset):
    def __init__(
        self,
        root,
        download=False,
    ):
        self.resources = [
            "http://press.liacs.nl/mirflickr/mirflickr1m.v3b/features_edgehistogram.zip",
            "http://press.liacs.nl/mirflickr/mirflickr1m.v3b/features_homogeneoustexture.zip"
            "http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip",
            "http://press.liacs.nl/mirflickr/mirflickr1m.v3b/tags.zip",
        ]
        self.root = root
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def __getitem__(self, index):
        batch = {"index": index}
        image, label = self.images[index // 10]
        batch["image"] = image
        batch["label"] = label
        batch["sentence"] = self.sentences[index][0]
        return batch

    def __len__(self):
        return len(self.sentences)

    def _check_raw_exists(self) -> bool:
        return (
            os.path.exists(
                os.path.join(
                    self.raw_folder,
                    "features_edgehistogram.zip",
                )
            )
            and os.path.exists(
                os.path.join(
                    self.raw_folder,
                    "features_homogeneoustexture.zip",
                )
            )
            and os.path.exists(
                os.path.join(
                    self.raw_folder,
                    "mirflickr25k_annotations_v080.zip",
                )
            )
        )

    def _check_exists(self) -> bool:
        return (
            os.path.exists(
                os.path.join(
                    self.raw_folder,
                    "features_edgehistogram",
                )
            )
            and os.path.exists(
                os.path.join(
                    self.raw_folder,
                    "features_homogeneoustexture",
                )
            )
            and os.path.exists(
                os.path.join(
                    self.raw_folder,
                    "mirflickr25k_annotations_v080",
                )
            )
        )

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
