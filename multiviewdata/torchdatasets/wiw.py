import math
import os
import zipfile

import h5py
import numpy as np
from torch.utils.data import Dataset


class WIWDataset(Dataset):
    def __init__(self, root, feats=None, partials=None, split="train", download=True):
        """

        :param root: Root directory of dataset
        :param feats: Which features to use from ["eng", "ru", "ger", "it", "vis"]
        :param partials: Which features to use as partials from ["eng", "ru", "ger", "it", "vis"]
        :param split:
        :param download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        """
        self.resources = [
            (
                "https://mega.nz/file/Gc0kHBTA#CYpHo_Vs2j1BIML2rlBhxFtOzzAzpkhSIeYT3rE93Go",
                "wiw_data.zip",
            )
        ]
        self.root = root
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )
        # Dataset parameters

        self.lng_dict = {
            "eng": "english",
            "ger": "german",
            "it": "italian",
            "ru": "russian",
        }
        if feats is None:
            self.feats = ["eng", "ru"]
        if partials is None:
            self.partials = ["vis"]
        self.folder = self.feats[0] + "_" + self.feats[1]
        self.dname = "/wiw_" + self.folder + "_img_pca_300_wv_splitted"  # Dataset Name
        self.dataset = h5py.File(
            os.path.join(self.raw_folder, "wiw_data/")
            + self.folder
            + "/"
            + self.dname
            + ".h5",
            "r",
        )[split]
        self.comment = ""

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.raw_folder, "wiw_data"))

    def _check_raw_exists(self) -> bool:
        return os.path.exists(os.path.join(self.raw_folder, "wiw_data.zip"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        batch = {"index": index.astype(np.float32)}
        batch["views"] = [
            self.dataset["%06d" % index][feat + "_feats"][()].astype(np.float32)
            for feat in self.feats
        ]
        if self.partials is not None:
            batch["partials"] = [
                self.dataset["%06d" % index][partial + "_feats"][()].astype(np.float32)
                for partial in self.partials
            ]
        return batch

    def download(self) -> None:
        """Download the data if it doesn't exist in processed_folder already."""

        if not self._check_raw_exists():
            os.makedirs(self.raw_folder, exist_ok=True)
            print(
                f"Download raw zip file from https://mega.nz/file/eV0STDTR#w6Xg248RQdzL28VOmoqsFLidJqrlSZKx7f8AGqfA204"
                f"and put it in {self.raw_folder}. This is manual because the python api is extremely slow."
            )
        if self._check_exists():
            return
        with zipfile.ZipFile(os.path.join(self.raw_folder, "wiw_data.zip")) as z:
            z.extractall(self.raw_folder)
        print("Processing...")
        self.process()
        print("Done!")

    def process(self):
        for languages in [("eng", "ger"), ("eng", "it"), ("eng", "ru")]:
            # Split data
            lng1, lng2 = languages
            dataset_name = (
                os.path.join(self.raw_folder, "wiw_data/")
                + lng1
                + "_"
                + lng2
                + "/wiw_"
                + lng1
                + "_"
                + lng2
                + "_img_pca_300_wv.h5"
            )
            dataname = h5py.File(dataset_name, "r")
            dataset_split = dataname["train"]

            samples_size = count_split_size(dataset_split)
            train_size = int(math.ceil(0.7 * samples_size))
            val_size = int(math.ceil(0.15 * samples_size))
            test_size = samples_size - (train_size + val_size)
            splits_size = [train_size, val_size, test_size]

            train_data = []
            val_data = []
            test_data = []

            for idx in range(samples_size):
                tup = dataset_split["%06d" % idx]
                lng1_feats = tup[lng1 + "_feats"][()]
                lng2_feats = tup[lng2 + "_feats"][()]
                vis_feats = tup["vis_feats"][()]
                lng1_desc = tup[lng1 + "_descriptions_feats"][()][0]  # .value[0]
                lng2_desc = tup[lng2 + "_descriptions_feats"][()][0]  # .value[0]

                if idx < train_size:
                    train_data.append(
                        (lng1_feats, lng2_feats, vis_feats, lng1_desc, lng2_desc)
                    )
                elif idx < train_size + val_size:
                    val_data.append(
                        (lng1_feats, lng2_feats, vis_feats, lng1_desc, lng2_desc)
                    )
                else:
                    test_data.append(
                        (lng1_feats, lng2_feats, vis_feats, lng1_desc, lng2_desc)
                    )

            # Write data
            fname = dataset_name[:-3] + "_splitted"
            h5output = h5py.File(fname + ".h5", "w")
            # The HDF5 file will contain a top-level group for each split
            train = h5output.create_group("train")
            val = h5output.create_group("val")
            test = h5output.create_group("test")
            splits = ["train", "val", "test"]
            split_dims = []
            img_dims = train_data[0][2].shape[0]
            txt_dims = [train_data[0][0].shape[0], train_data[0][1].shape[0]]
            data_idx = 0
            for split in splits:
                dims = splits_size[data_idx]
                textual_features = []
                data_dim_idx = 0
                for dim_idx in range(dims):
                    if split == "train":
                        container = train.create_group("%06d" % data_dim_idx)
                    elif split == "val":
                        container = val.create_group("%06d" % data_dim_idx)
                    else:
                        container = test.create_group("%06d" % data_dim_idx)
                    lng_idx = 0
                    if split == "train":
                        for j, lng in enumerate(languages):
                            text_data = container.create_dataset(
                                lng + "_feats", (txt_dims[j],), dtype="float32"
                            )
                            text_data[:] = train_data[dim_idx][j]
                            text_data = container.create_dataset(
                                lng + "_descriptions_feats",
                                (1,),
                                dtype=h5py.special_dtype(vlen=str),
                            )
                            text_data[:] = train_data[dim_idx][3 + j]
                            lng_idx += 1
                        image_data = container.create_dataset(
                            "vis_feats", (img_dims,), dtype="float32"
                        )
                        image_data[:] = train_data[dim_idx][2]
                    elif split == "val":
                        for j, lng in enumerate(languages):
                            text_data = container.create_dataset(
                                lng + "_feats", (txt_dims[j],), dtype="float32"
                            )
                            text_data[:] = val_data[dim_idx][j]
                            text_data = container.create_dataset(
                                lng + "_descriptions_feats",
                                (1,),
                                dtype=h5py.special_dtype(vlen=str),
                            )
                            text_data[:] = val_data[dim_idx][3 + j]
                            lng_idx += 1
                        image_data = container.create_dataset(
                            "vis_feats", (img_dims,), dtype="float32"
                        )
                        image_data[:] = val_data[dim_idx][2]
                    else:
                        for j, lng in enumerate(languages):
                            text_data = container.create_dataset(
                                lng + "_feats", (txt_dims[j],), dtype="float32"
                            )
                            text_data[:] = test_data[dim_idx][j]
                            text_data = container.create_dataset(
                                lng + "_descriptions_feats",
                                (1,),
                                dtype=h5py.special_dtype(vlen=str),
                            )
                            text_data[:] = test_data[dim_idx][3 + j]
                            lng_idx += 1
                        image_data = container.create_dataset(
                            "vis_feats", (img_dims,), dtype="float32"
                        )
                        image_data[:] = test_data[dim_idx][2]
                    data_dim_idx += 1
                data_idx += 1
            h5output.close()


def count_split_size(dataset_split):
    split_size = 0
    for _ in dataset_split:
        split_size += 1
    return split_size
