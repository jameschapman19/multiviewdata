import itertools
from typing import List, Union

import numpy as np
from scipy import linalg
from scipy.linalg import block_diag
from sklearn.utils.validation import check_random_state
from ..utils import _process_parameter


def generate_covariance_data(
    n: int,
    view_features: List[int],
    latent_dims: int = 1,
    view_sparsity: List[Union[int, float]] = None,
    correlation: Union[List[float], float] = 1,
    structure: Union[str, List[str]] = None,
    sigma: Union[List[float], float] = None,
    decay: float = 0.5,
    positive=None,
    random_state: Union[int, np.random.RandomState] = None,
):
    """
    Function to generate CCA dataset with defined population correlations

    :param n: number of samples
    :param view_sparsity: level of sparsity in features in each view either as number of active variables or percentage active
    :param view_features: number of features in each view
    :param latent_dims: number of latent dimensions
    :param correlation: correlation either as list with element for each latent dimension or as float which is scaled by 'decay'
    :param structure: within view covariance structure ('identity','gaussian','toeplitz','random')
    :param sigma: gaussian sigma
    :param decay: ratio of second signal to first signal
    :return: tuple of numpy arrays: view_1, view_2, true weights from view 1, true weights from view 2, overall covariance structure

    :Example:

    """
    random_state = check_random_state(random_state)
    structure = _process_parameter(
        "structure", structure, "identity", len(view_features)
    )
    view_sparsity = _process_parameter(
        "view_sparsity", view_sparsity, 1, len(view_features)
    )
    positive = _process_parameter("positive", positive, False, len(view_features))
    sigma = _process_parameter("sigma", sigma, 0.5, len(view_features))
    completed = False
    attempts = 0
    while not completed and attempts < 10:
        try:
            mean = np.zeros(sum(view_features))
            if not isinstance(correlation, list):
                p = np.arange(0, latent_dims)
                correlation = correlation * decay**p
            covs = []
            true_features = []
            for view_p, sparsity, view_structure, view_positive, view_sigma in zip(
                view_features, view_sparsity, structure, positive, sigma
            ):
                # Covariance Bit
                if view_structure == "identity":
                    cov_ = np.eye(view_p)
                elif view_structure == "gaussian":
                    cov_ = _generate_gaussian_cov(view_p, view_sigma)
                elif view_structure == "toeplitz":
                    cov_ = _generate_toeplitz_cov(view_p, view_sigma)
                elif view_structure == "random":
                    cov_ = _generate_random_cov(view_p, random_state)
                else:
                    completed = True
                    print("invalid structure")
                    break
                weights = random_state.randn(view_p, latent_dims)
                if sparsity <= 1:
                    sparsity = np.ceil(sparsity * view_p).astype("int")
                if sparsity < view_p:
                    mask = np.stack(
                        (
                            np.concatenate(
                                ([0] * (view_p - sparsity), [1] * sparsity)
                            ).astype(bool),
                        )
                        * latent_dims,
                        axis=0,
                    ).T
                    mask = mask.flatten()
                    random_state.shuffle(mask)
                    mask = mask.reshape(weights.shape)
                    weights = weights * mask
                    if view_positive:
                        weights[weights < 0] = 0
                weights = _decorrelate_dims(weights, cov_)
                weights /= np.sqrt(np.diag((weights.T @ cov_ @ weights)))
                true_features.append(weights)
                covs.append(cov_)

            cov = block_diag(*covs)

            splits = np.concatenate(([0], np.cumsum(view_features)))

            for i, j in itertools.combinations(range(len(splits) - 1), 2):
                cross = np.zeros((view_features[i], view_features[j]))
                for _ in range(latent_dims):
                    A = correlation[_] * np.outer(
                        true_features[i][:, _], true_features[j][:, _]
                    )
                    # Cross Bit
                    cross += covs[i] @ A @ covs[j]
                cov[
                    splits[i] : splits[i] + view_features[i],
                    splits[j] : splits[j] + view_features[j],
                ] = cross
                cov[
                    splits[j] : splits[j] + view_features[j],
                    splits[i] : splits[i] + view_features[i],
                ] = cross.T

            X = np.zeros((n, sum(view_features)))
            chol = np.linalg.cholesky(cov)
            for _ in range(n):
                X[_, :] = _chol_sample(mean, chol, random_state)
            views = np.split(X, np.cumsum(view_features)[:-1], axis=1)
            completed = True
        except:
            completed = False
            attempts += 1
    if not completed:
        raise ValueError(
            "Could not generate positive definite covariance matrix for supplied parameters."
        )
    return views, true_features


def _decorrelate_dims(up, cov):
    A = up.T @ cov @ up
    for k in range(1, A.shape[0]):
        up[:, k:] -= np.outer(up[:, k - 1], A[k - 1, k:] / A[k - 1, k - 1])
        A = up.T @ cov @ up
    return up


def _chol_sample(mean, chol, random_state):
    rng = check_random_state(random_state)
    return mean + chol @ rng.randn(mean.size)


def _gaussian(x, mu, sig, dn):
    """
    Generate a gaussian covariance matrix

    :param x:
    :param mu:
    :param sig:
    :param dn:
    """
    return (
        np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))
        * dn
        / (np.sqrt(2 * np.pi) * sig)
    )


def _generate_gaussian_cov(p, sigma):
    x = np.linspace(-1, 1, p)
    x_tile = np.tile(x, (p, 1))
    mu_tile = np.transpose(x_tile)
    dn = 2 / (p - 1)
    cov = _gaussian(x_tile, mu_tile, sigma, dn)
    cov /= cov.max()
    return cov


def _generate_toeplitz_cov(p, sigma):
    c = np.arange(0, p)
    c = sigma**c
    cov = linalg.toeplitz(c, c)
    return cov


def _generate_random_cov(p, random_state):
    rng = check_random_state(random_state)
    cov_ = rng.randn(p, p)
    U, S, Vt = np.linalg.svd(cov_.T @ cov_)
    cov = U @ (1 + np.diag(rng.randn(p))) @ Vt
    return cov
