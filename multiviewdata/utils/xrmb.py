from scipy.io import loadmat


def load_xrmb():
    """
    Download, parse and process xrmb data
    Examples
    --------


    Returns
    -------
    train_view_1, train_view_2, test_view_1, test_view_2
    """
    view_1 = loadmat(datadir + "XRMBf2KALDI_window7_single1.mat")
    view_2 = loadmat(datadir + "XRMBf2KALDI_window7_single2.mat")

    return view_1["X1"], view_2["X2"], view_1["XTe1"], view_2["XTe2"]
