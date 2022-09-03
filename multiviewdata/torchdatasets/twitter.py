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

from multiviewdata.torchdatasets.utils import fopen


class Twitter(Dataset):
    """
    Learning Multiview Representations of Twitter Users
    ----------------------------------------------------

    Input views used to learn multiview Twitter user embeddings

    Twitter's terms of service prevents sharing of large scale Twitter corpora. Instead, we share the
      1000-dimensional PCA vectors produced for each user's tweet and network views. These embeddings can be used in place of the user data to reproduce our methods and to compare new methods against our work.

    One row per user, tab-delimited
    First field is Twitter user ID
    Next 6 fields at indicator features for whether this view contains data for this specific user
    The final 6 fields are views, each containing a 1000-dimensional space-delimited vector

    Format
    ------

    The data file contains these fields in tab separated format:

      UserID
      EgoTweets
      MentionTweets
      FriendTweets
      FollowerTweets
      FriendNetwork
      FollowerNetwork

    Vector dimensions are sorted in order of decreasing variance, so evaluating
    a 50-dimensional PCA vector means just using the first 50 values in each view.

    User IDs for user engagement and friend prediction tasks

    Each row in a file corresponds to a single hashtag or celebrity.  The first field is the hashtag users posted or celebrity they follow.  All following entries are the user IDs of everyone who engaged.  The first 10 user IDs were used to compute the query embedding (rank all other user IDs by cosine similarity).  Hashtags are split into development and test, as used in the paper.

    Parameters
    ----------
    root : string
        Root directory of dataset where directory
        ``twitter`` exists or will be saved to if download is set to True.
    viewstokeep : list of ints
        Which views to keep. 0-5 are the views, 6-11 are the indicators
    download : bool, default=False
        If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.
    maxrows : int, default=-1
        Maximum number of rows to load. -1 means all rows
    replaceempty : bool, default=True
        If true, replace empty views with a random vector

    """
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
        batch["userid"] = self.ids[index].astype(np.float32)
        batch["views"] = [view[index].astype(np.float32) for view in self.views]
        return batch

    def __len__(self):
        return len(self.dataset["views"][0])

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
