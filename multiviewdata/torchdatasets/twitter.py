"""
Processing of Twitter Data:

MIT License

Copyright (c) 2016 Adrian Benton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

import gzip
import os

import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_and_extract_archive


class TwitterDataset(Dataset):
    def __init__(
        self, root, viewstokeep=None, download=False, maxrows=-1, replaceempty=True
    ):
        self.resources = [
            "https://www.cs.jhu.edu/~mdredze/datasets/multiview_embeddings/user_6views_tfidf_pcaEmbeddings_userTweets+networks.tsv.gz",
            "https://www.cs.jhu.edu/~mdredze/datasets/multiview_embeddings/friend_and_hashtag_prediction_userids.zip",
        ]
        self.root = root
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        ids, views = ldViews(
            os.path.join(
                self.raw_folder,
                "user_6views_tfidf_pcaEmbeddings_userTweets+networks.tsv",
            ),
            viewstokeep,
            replaceempty=replaceempty,
            maxrows=maxrows,
        )
        self.dataset = dict(ids=ids, views=views)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def __getitem__(self, index):
        batch = {"index": index}
        batch["userid"] = self.ids[index]
        batch["views"] = [view[index] for view in self.views]
        return batch

    def __len__(self):
        return self.v1.shape[0]

    def _check_raw_exists(self) -> bool:
        return os.path.exists(
            os.path.join(
                self.raw_folder,
                "user_6views_tfidf_pcaEmbeddings_userTweets+networks.tsv.gz",
            )
        ) and os.path.exists(
            os.path.join(self.raw_folder, "friend_and_hashtag_prediction_userids.zip")
        )

    def _check_exists(self) -> bool:
        return os.path.exists(
            os.path.join(
                self.raw_folder,
                "user_6views_tfidf_pcaEmbeddings_userTweets+networks.tsv",
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


def fopen(p, flag="r"):
    """Opens as gzipped if appropriate, else as ascii."""
    if p.endswith(".gz"):
        if "w" in flag:
            return gzip.open(p, "wb")
        else:
            return gzip.open(p, "rt")
    else:
        return open(p, flag)


def ldViews(inPath, viewstokeep, replaceempty=True, maxrows=-1):
    """
    Loads dataset for each view.
    Input file is tab-separated: first column ID, next k columns are number of documents that
    go into each view, other columns are the views themselves.  Features in each view are
    space-separated floats -- dense.

    replaceempty: If true, then replace empty rows with the mean embedding for each view

    Returns list of IDs, and list with an (N X F) matrix for each view.
    """

    N = 0  # Number of examples

    V = 0  # Number of views
    FperV = []  # Number of features per view

    f = fopen(inPath)
    flds = f.readline().split("\t")

    V = int((len(flds) - 1) / 2)

    # Use all views
    if not viewstokeep:
        viewstokeep = [i for i in range(V)]

    for fld in flds[1 + V :]:
        FperV.append(len(fld.split()))
    f.close()

    f = fopen(inPath)

    flds = f.readline().strip().split("\t")

    F = [len(fld.split()) for fld in flds[(1 + V) :]]
    N += 1

    # Only reduce to kept views
    F = [F[v] for v in viewstokeep]

    for ln in f:
        N += 1
    f.close()

    data = [np.zeros((maxrows, numFs)) for numFs in F]
    ids = []

    f = fopen(inPath)
    for lnidx, ln in enumerate(f):
        if (maxrows > 0) and (lnidx >= maxrows):
            break

        if not lnidx % 10000:
            print("Reading line: %dK" % (lnidx / 1000))

        flds = ln.split("\t")
        ids.append(int(flds[0]))

        viewstrs = flds[(1 + V) :]

        for idx, viewstr in enumerate(viewstrs):
            if idx not in viewstokeep:
                continue

            idx = viewstokeep.index(idx)

            for fidx, v in enumerate(viewstr.split()):
                data[idx][lnidx, fidx] = float(v)

    f.close()

    # Replace empty rows with the mean for each view
    if replaceempty:
        means = [
            np.sum(d, axis=0) / np.sum(1.0 * (np.abs(d).sum(axis=1) > 0.0))
            for d in data
        ]
        for i in range(maxrows):
            for nvIdx, vIdx in enumerate(viewstokeep):
                if np.sum(data[nvIdx][i, :]) == 0.0:
                    data[nvIdx][i, :] = means[nvIdx]

    return ids, data


def ldK(p, viewstokeep):
    """
    Returns matrix K, indicating which (example, view) pair is missing.

    p: Path to data file
    viewstokeep: Indices of views we want to keep.
    """

    numLns = 0

    f = fopen(p)
    for ln in f:
        numviews = ln.count("\t") / 2
        numLns += 1
    f.close()

    # Use all views
    if not viewstokeep:
        viewstokeep = [i for i in range(numviews)]

    K = np.ones((numLns, len(viewstokeep)))

    # We keep count of number of tweets and infos collected in each view, and first field is the user ID.
    # Just zero out those views where we did not collect any data for that view
    f = fopen(p)
    for lnIdx, ln in enumerate(f):
        flds = ln.split("\t")
        for idx, vidx in enumerate(viewstokeep):
            count = int(flds[vidx + 1])
            if count < 1:
                K[lnIdx, idx] = 0.0
    f.close()

    return K
