import numpy as np
from tqdm import tqdm
import multiprocessing
import ctypes
from functools import partial
import scipy


class CosineDistance:
    def __init__(self, T=1) -> None:
        self.T = T

    def __call__(self, mu, sigma, z):
        """
        z: n x num_z_samples x 512
        sigma: K x 512
        mu: K x 512
        """
        z = np.moveaxis(z, 2, 1)  # n x 512 x num_z_samples
        a_ilj_final = mu @ z
        assert a_ilj_final.shape[0] == z.shape[0]
        a_ilj_final = np.moveaxis(a_ilj_final, 2, 1)  # n x num_z_samples x K
        return a_ilj_final / self.T


class DefaultSCF:
    def __init__(self, T=1, kappa_shift=0) -> None:
        self.T = T
        self.kappa_shift = kappa_shift

    def __call__(self, mu, kappa, z):
        """
        z: n x num_z_samples x 512
        mu: K x 512
        kappa: K
        """
        mu = mu.astype(np.float64)
        kappa = kappa.astype(np.float64)
        z = z.astype(np.float64)
        kappa = kappa + self.kappa_shift

        d = z.shape[-1]  # 512
        z = np.moveaxis(z, 2, 1)  # n x 512 x num_z_samples
        a_ilj = mu @ z
        a_ilj = np.moveaxis(a_ilj, 2, 1)  # n x num_z_samples x K

        log_ive_j = np.log(
            1e-6 + scipy.special.ive(d / 2 - 1, kappa, dtype=kappa.dtype)
        )  # K
        log_kappa_j = np.log(1e-6 + kappa)  # K

        a_ilj = (
            kappa[np.newaxis, np.newaxis, :] * (a_ilj - 1)
            + (d / 2 - 1) * log_kappa_j[np.newaxis, np.newaxis, :]
            - (d / 2) * np.log(2 * np.pi)
            - log_ive_j[np.newaxis, np.newaxis, :]
        )

        a_ilj_final = a_ilj / self.T

        return a_ilj_final


class DefaultPfe:
    def __init__(self, T=1, sigma_shift=0) -> None:
        self.T = T
        self.sigma_shift = sigma_shift

    def __call__(self, mu, sigma, z):
        """
        z: n x num_z_samples x 512
        sigma: K x 512
        mu: K x 512
        """
        mu = mu.astype(np.float64)
        sigma = sigma.astype(np.float64)
        z = z.astype(np.float64)
        sigma = sigma + self.sigma_shift

        b = np.sum(np.log(sigma), axis=1)  # K
        z = np.moveaxis(z, 2, 1)  # n x 512 x num_z_samples
        m = -2 * ((mu / sigma) @ z)  # n x K x num_z_samples
        z_sq = z**2
        t = (1 / sigma) @ z_sq  # n x K x num_z_samples
        n = np.sum(mu**2 / sigma, axis=1)  # K
        a_ilj_final = (
            b[np.newaxis, :, np.newaxis] + t + m + n[np.newaxis, :, np.newaxis]
        )
        a_ilj_final = -0.5 * np.moveaxis(a_ilj_final, 2, 1) / self.T
        return a_ilj_final


def cos_pfe_likelihood(mu, sigma, z):
    """
    z: n x num_z_samples x 512
    sigma: K x 512
    mu: K x 512
    """
    a_ilj_final = np.zeros(shape=z.shape[:-1] + mu.shape[:-1])  # placeholder
    num_dims = mu.shape[1]
    for k in tqdm(range(num_dims)):
        a_ilj = (z[:, :, np.newaxis, k] * mu[np.newaxis, np.newaxis, :, k]) / sigma[
            np.newaxis, np.newaxis, :, k
        ]  # - np.log(sigma)[np.newaxis, np.newaxis, :, k]
        np.add(a_ilj_final, a_ilj, out=a_ilj_final)

    a_ilj_final = 0.5 * a_ilj_final
    return a_ilj_final
