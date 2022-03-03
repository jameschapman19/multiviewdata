import os


def test_noisymnist():
    from multiviewdata.torchdatasets import NoisyMNISTDataset
    a = NoisyMNISTDataset(os.getcwd(), download=True)[0]
    assert "index" in a
    assert "views" in a

def test_tangledmnist():
    from multiviewdata.torchdatasets import TangledMNISTDataset
    a = TangledMNISTDataset(os.getcwd(), download=True)[0]
    assert "index" in a
    assert "views" in a

def test_splitmnist():
    from multiviewdata.torchdatasets import SplitMNISTDataset
    a = SplitMNISTDataset(os.getcwd(), download=True)[0]
    assert "index" in a
    assert "views" in a

def test_mfeat():
    from multiviewdata.torchdatasets import MFeatDataset
    a = MFeatDataset(os.getcwd(), download=True)[0]
    assert "index" in a
    assert "views" in a