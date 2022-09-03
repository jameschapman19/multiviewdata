import os
import urllib

import numpy as np
from torch.utils.data.dataset import Dataset


class MediaMill(Dataset):
    """
    Class to generate paired noisy mnist data
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        download=False,
    ):
        """
        :param root: Root directory of dataset
        :param train: whether this is train or test
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
        input_size = 120
        target_size = 101

        def convert_target(target_str):
            targets = np.zeros((target_size))
            if target_str != '':
                for l in target_str.split(','):
                    id = int(l)
                    targets[id] = 1
            return targets

        train_file, valid_file, test_file = [os.path.join(root, 'mediamill_' + ds + '.libsvm') for ds in
                                             ['train', 'valid', 'test']]

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def __getitem__(self, index):
        batch = {"index": index}
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
            urllib.request.urlretrieve(
                'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/mediamill/train-exp1.svm.bz2',
                os.path.join(self.raw_folder, 'train-exp1.svm.bz2'))
            urllib.request.urlretrieve(
                'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/mediamill/test-exp1.svm.bz2',
                os.path.join(self.raw_folder, 'test-exp1.svm.bz2'))
            import bz2
            train_valid_bz2_file = bz2.BZ2File(os.path.join(self.raw_folder, 'train-exp1.svm.bz2'))
            test_bz2_file = bz2.BZ2File(os.path.join(self.raw_folder, 'test-exp1.svm.bz2'))
            train_file, valid_file, test_file = [
                open(os.path.join(self.raw_folder, 'mediamill_' + ds + '.libsvm'), 'w') for ds
                in ['train', 'valid', 'test']]

            # Putting train/valid data in memory
            train_valid_data = [line for line in train_valid_bz2_file]
            line_id = 0
            train_valid_split = 25828
            for i in range(len(train_valid_data)):
                s = train_valid_data[i]
                if line_id < train_valid_split:
                    train_file.write(s)
                else:
                    valid_file.write(s)
                line_id += 1
            train_file.close()
            valid_file.close()
            train_valid_bz2_file.close()

        for line in test_bz2_file:
            test_file.write(line)

        test_file.close()
        test_bz2_file.close()
        return

if __name__ == '__main__':
    MediaMill(root='',download=True)
