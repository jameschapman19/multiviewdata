from typing import List, Union

import numpy as np
from sklearn.utils.validation import check_random_state
from ..utils import _process_parameter


def generate_simple_data(
    n: int,
    view_features: List[int],
    view_sparsity: List[Union[int, float]] = None,
    eps: float = 0,
    transform=False,
    random_state=None,
):
    """
    Simple latent variable model to generate data with one latent factor

    :param n: number of samples
    :param view_features: number of features view 1
    :param view_sparsity: number of features view 2
    :param eps: gaussian noise std
    :return: view1 matrix, view2 matrix, true weights view 1, true weights view 2

    :Example:

    """
    random_state = check_random_state(random_state)
    z = random_state.randn(n)
    if transform:
        z = np.sin(z)
    views = []
    true_features = []
    view_sparsity = _process_parameter(
        "view_sparsity", view_sparsity, 0, len(view_features)
    )
    for p, sparsity in zip(view_features, view_sparsity):
        weights = random_state.normal(size=(p, 1))
        if sparsity <= 1:
            sparsity = np.ceil(sparsity * p).astype("int")
            weights[random_state.choice(np.arange(p), p - sparsity, replace=False)] = 0
        gaussian_x = random_state.normal(0, eps, size=(n, p)) * eps
        view = np.outer(z, weights)
        view += gaussian_x
        views.append(view / np.linalg.norm(view, axis=0))
        true_features.append(weights)
    return views, true_features
