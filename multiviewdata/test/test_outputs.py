import os


def test_noisymnist():
    from multiviewdata.torchdatasets import NoisyMNIST
    a = NoisyMNIST(os.getcwd(), download=True)[0]
    assert "index" in a
    assert "views" in a

def test_tangledmnist():
    from multiviewdata.torchdatasets import TangledMNIST
    a = TangledMNIST(os.getcwd(), download=True)[0]
    assert "index" in a
    assert "views" in a

def test_splitmnist():
    from multiviewdata.torchdatasets import SplitMNIST
    a = SplitMNIST(os.getcwd(), download=True)[0]
    assert "index" in a
    assert "views" in a

def test_mfeat():
    from multiviewdata.torchdatasets import MFeat
    a = MFeat(os.getcwd(), download=True)[0]
    assert "index" in a
    assert "views" in a
