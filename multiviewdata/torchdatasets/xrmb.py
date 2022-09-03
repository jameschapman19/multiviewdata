import os

from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import numpy as np


class XRMB(Dataset):
    """
    Parameters
    ----------
    root : string
        Root directory of dataset where directory
        ``XRMBf2KALDI_window7_single1.mat`` exists.
    train : bool, optional
        If True, creates dataset from training set, otherwise
        creates from test set.
    download : bool, optional
        If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.

    References
    ----------

    The original XRMB manual can be found here:
    http://www.haskins.yale.edu/staff/gafos_downloads/ubdbman.pdf \n\n We acknowledge John Westbury for providing
    the original data and for permitting this post-processed version to be redistributed. The original data
    collection was supported (in part) by research grant number R01 DC 00820 from the National Institute of
    Deafness and Other Communicative Disorders, U.S. National Institutes of Health. \n\nThe post-processed data
    provided here was produced as part of work supported in part by NSF grant IIS-1321015.\n\nSome of the
    original XRMB articulatory data was missing due to issues such as pellet tracking errors.  The data has been
    reconstructed in using the algorithm described in this paper: \n\n Wang, Arora, and Livescu, Reconstruction
    of articulatory measurements with smoothed low-rank matrix completion, SLT 2014. \n\n
    http://ttic.edu/livescu/papers/wang_SLT2014.pdf \n\n The data provided here has been used for multi-view
    acoustic feature learning in this paper:\n\nWang, Arora, Livescu, and Bilmes, Unsupervised learning of
    acoustic features via deep canonical correlation analysis, ICASSP
    2015.\n\nhttp://ttic.edu/livescu/papers/wang_ICASSP2015.pdf \n\n If you use this version of the data,
    please cite the papers above.
    """
    def __init__(
        self,
        root,
        train=True,
        download=False,
    ):
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
        view_1_file = os.path.join(self.raw_folder, "XRMBf2KALDI_window7_single1.mat")
        view_2_file = os.path.join(self.raw_folder, "XRMBf2KALDI_window7_single2.mat")
        view_1, view_2 = (
            loadmat(view_1_file),
            loadmat(view_2_file),
        )
        if train:
            view_1=view_1["X1"]
            view_2=view_2["X2"]
        else:
            view_1 = view_1["XTe1"]
            view_2 = view_2["XTe2"]
        self.dataset = dict(view_1=view_1, view_2=view_2)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def __len__(self):
        return len(self.dataset["view_1"])

    def __getitem__(self, index):
        return {
            "views": (
                self.dataset["view_1"][index].astype(np.float32),
                self.dataset["view_2"][index].astype(np.float32),
            ),
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
