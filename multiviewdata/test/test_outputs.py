import os

from multiviewdata.torchdatasets import NoisyMNISTDataset


def test_noisymnist():
    a = NoisyMNISTDataset(os.getcwd(), download=True)[0]
    assert "index" in a
    assert "views" in a
