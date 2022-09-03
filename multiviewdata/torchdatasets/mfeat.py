import os

import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_and_extract_archive


class MFeat(Dataset):
    def __init__(
        self,
        root: str,
        feats: list = None,
        partials: list = None,
        download: bool = False,
    ):
        """
        The multi-feature digit dataset
        -------------------------------

        Owned and donated by:
        ----------------------

        Robert P.W. Duin
        Department of Applied Physics
        Delft University of Technology
        P.O. Box 5046, 2600 GA Delft
        The Netherlands

        email: duin@ph.tn.tudelft.nl
        http : //www.ph.tn.tudelft.nl/~duin
        tel +31 15 2786143

        Usage
        -----
        A slightly different version of the database is used in

        M. van Breukelen, R.P.W. Duin, D.M.J. Tax, and J.E. den Hartog, Handwritten
             digit recognition by combined classifiers, Kybernetika, vol. 34, no. 4,
             1998, 381-386.

        M. van Breukelen and R.P.W. Duin, Neural Network Initialization by Combined
             Classifiers, in: A.K. Jain, S. Venkatesh, B.C. Lovell (eds.), ICPR'98,
             Proc. 14th Int. Conference on Pattern Recognition (Brisbane, Aug. 16-20),

        The database as it is is used in:

        A.K. Jain, R.P.W. Duin, J. Mao, Statisitcal Pattern Recognition: A Review,
             in preparation

        Description
        -----------

        This dataset consists of features of handwritten numerals (`0'--`9')
        extracted from a collection of Dutch utility maps. 200 patterns per
        class (for a total of 2,000 patterns) have been digitized in  binary
        images. These digits are represented in terms of the following six
        feature sets (files):

        1. mfeat-fou: 76 Fourier coefficients of the character shapes;
        2. mfeat-fac: 216 profile correlations;
        3. mfeat-kar: 64 Karhunen-Lo�ve coefficients;
        4. mfeat-pix: 240 pixel averages in 2 x 3 windows;
        5. mfeat-zer: 47 Zernike moments;
        6. mfeat-mor: 6 morphological features.

        In each file the 2000 patterns are stored in ASCI on 2000 lines. The
        first 200 patterns are of class `0', followed by sets of 200 patterns
        for each of the classes `1' - `9'. Corresponding patterns in different
        feature sets (files) correspond to the same original character.

        The source image dataset is lost. Using the pixel-dataset (mfeat-pix)
        sampled versions of the original images may be obtained (15 x 16 pixels).

        Total number of instances:
        --------------------------
        2000 (200 instances per class)

        Total number of attributes:
        ---------------------------
        649 (distributed over 6 datasets,see above)

        no missing attributes

        Total number of classes:
        ------------------------
        10

        Format:
        ------
        6 files, see above.
        Each file contains 2000 lines, one for each instance.
        Attributes are SPACE separated and can be loaded by Matlab as
        > load filename
        No missing attributes. Some are integer, others are real.

        Parameters
        ----------
        root : str
            Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        feats : list
            List of features to load. If None, all features are loaded.
        partials : list
            List of partial features to load. If None, no partial features are loaded.
        download : bool, optional
            If true, downloads the dataset from the internet and
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
        batch["views"] = [
            self.dataset[feat][index].astype(np.float32) for feat in self.feats
        ]
        if self.partials is not None:
            batch["partials"] = [
                self.dataset[partial][index].astype(np.float32)
                for partial in self.partials
            ]
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
