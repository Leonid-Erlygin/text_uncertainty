import numpy as np

from .distance_uncertainty_funcs import l2_normalize

from training.evaluation import l2_normalize
from scipy.special import softmax


def aggregate_PFE(x, sigma_sq=None, normalize=True, concatenate=False):
    if sigma_sq is None:
        D = int(x.shape[1] / 2)
        mu, sigma_sq = x[:, :D], x[:, D:]
    else:
        mu = x
    attention = 1.0 / sigma_sq
    attention = attention / np.sum(attention, axis=0, keepdims=True)

    mu_new = np.sum(mu * attention, axis=0)
    sigma_sq_new = np.min(sigma_sq, axis=0)

    if normalize:
        mu_new = l2_normalize(mu_new)

    if concatenate:
        return np.concatenate([mu_new, sigma_sq_new])
    else:
        return mu_new, sigma_sq_new


def aggregate_SCF(x, kappa=None, normalize=True, concatenate=False):
    mu = x

    mu_new = np.sum(mu * kappa, axis=0) / np.sum(kappa)
    # kappa_new = np.max(kappa)
    kappa_new = np.mean(kappa)

    if normalize:
        mu_new = l2_normalize(mu_new)

    return mu_new, kappa_new


def aggregate_min(x, sigma_sq, normalize=True, concatenate=False):
    if sigma_sq is None:
        D = int(x.shape[1] / 2)
        mu, sigma_sq = x[:, :D], x[:, D:]
    else:
        mu = x

    best_features_indices = sigma_sq.argmin(axis=0)
    mu_new = mu[best_features_indices, range(x.shape[1])]
    sigma_sq_new = sigma_sq[best_features_indices, range(x.shape[1])]
    if normalize:
        mu_new = l2_normalize(mu_new)

    if concatenate:
        return np.concatenate([mu_new, sigma_sq_new])
    else:
        return mu_new, sigma_sq_new


def aggregate_softmax(x, sigma_sq, temperature=1.0, normalize=True, concatenate=False):
    if sigma_sq is None:
        D = int(x.shape[1] / 2)
        mu, sigma_sq = x[:, :D], x[:, D:]
    else:
        mu = x

    weights = np.exp(sigma_sq * temperature)
    weights /= weights.sum(axis=1, keepdims=True)
    mu_new = (mu * weights).sum(axis=0)

    if normalize:
        mu_new = l2_normalize(mu_new)

    return mu_new


def aggregate_templates(templates, method, normalize=True):
    for t in templates:
        if method == "first":
            t.mu = t.features[0]
            if normalize:
                t.mu = l2_normalize(t.mu)
            t.sigma_sq = t.sigmas[0]
        elif method == "SCF":
            t.mu, t.sigma_sq = aggregate_SCF(
                t.features, kappa=t.sigmas, normalize=normalize
            )
        elif method == "PFE":
            t.mu, t.sigma_sq = aggregate_PFE(
                t.features, sigma_sq=t.sigmas, normalize=normalize
            )
            # t.mu, t.sigma_sq = aggregate_PFE_properly(t.features, sigma_sq=t.sigmas) # bad aggregation
        elif method == "mean":
            t.mu = l2_normalize(np.mean(t.features, axis=0))
            t.sigma_sq = np.mean(t.sigmas, axis=0)
        elif method == "stat-mean":
            t.mu = l2_normalize(np.mean(t.features, axis=0))
            t.sigma_sq = np.mean(t.sigmas, axis=0) * (len(t.sigmas)) ** 0.5
        elif method == "argmax":
            idx = np.argmax(t.sigmas)
            t.mu = t.features[idx]
            t.sigma_sq = t.sigmas[idx]
        elif method == "stat-softmax":
            weights = softmax(t.sigmas[:, 0])
            t.mu = l2_normalize(np.dot(weights, t.features))
            t.sigma_sq = np.dot(weights, t.sigmas) * len(t.sigmas) ** 0.5
        elif method.startswith("softmax"):
            parts = method.split("-")
            if len(parts) == 1:
                temperature = 1.0
            else:
                temperature = float(parts[1])
            weights = softmax(t.sigmas[:, 0] / temperature)
            t.mu = l2_normalize(np.dot(weights, t.features))
            t.sigma_sq = np.dot(weights, t.sigmas)
        elif method == "weighted":
            mu = l2_normalize(t.features)
            weights = t.sigmas[:, 0]
            weights = weights / np.sum(weights)
            t.mu = l2_normalize(np.dot(weights, mu))
            t.sigma_sq = np.dot(weights, t.sigmas)
        elif method.startswith("weighted-softmax"):
            parts = method.split("-")
            if len(parts) == 2:
                temperature = 1.0
            else:
                temperature = float(parts[2])
            weights = t.sigmas[:, 0]
            weights = weights / np.sum(weights)
            t.mu = l2_normalize(np.dot(weights, t.features))
            weights = softmax(t.sigmas[:, 0] / temperature)
            t.sigma_sq = np.dot(weights, t.sigmas)
        elif method == "weighted":
            mu = l2_normalize(t.features)
            weights = t.sigmas[:, 0]
            weights = weights / np.sum(weights)
            t.mu = l2_normalize(np.dot(weights, mu))
            t.sigma_sq = np.dot(weights, t.sigmas)
            pass
        else:
            raise ValueError(f"Wrong aggregate method {method}")
