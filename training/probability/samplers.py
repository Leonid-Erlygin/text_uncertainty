import numpy as np


class MeanEstimate:
    def __init__(self, K) -> None:
        self.K = K
        assert K == 1

    def __call__(self, mu_verif, sigma_verif):
        """
        sigma: num_verif x 512
        mu: num_verif x 512

        return:
        Z with shape n x num_z_samples x 512
        z_weights num_z_samples
        """
        return mu_verif[:, np.newaxis, :], np.ones(1)


class HighSigma:
    def __init__(self, K, s, weights) -> None:
        self.K = K
        self.s = s
        self.weights = weights

    def __call__(self, mu_verif, sigma_verif):
        """
        sigma: num_verif x 512
        mu: num_verif x 512

        return:
        Z with shape n x num_z_samples x 512
        z_weights num_z_samples
        """
        d = sigma_verif.shape[1]
        num_verif = sigma_verif.shape[0]

        sigma_top_k_idx = np.flip(np.argsort(sigma_verif, axis=1), axis=1)[
            :, : self.K
        ]  # num_verif x K
        sigma_top_k_idx = sigma_top_k_idx[:, :, np.newaxis]  # num_verif x K x 1
        shift_vector = np.zeros((num_verif, self.K, d))
        shift_vector[
            np.arange(num_verif)[:, np.newaxis, np.newaxis],
            np.arange(self.K)[np.newaxis, :, np.newaxis],
            sigma_top_k_idx,
        ] = sigma_verif[
            np.arange(num_verif)[:, np.newaxis, np.newaxis], sigma_top_k_idx
        ]

        z_plus = mu_verif[:, np.newaxis, :] + self.s * shift_vector
        z_minus = mu_verif[:, np.newaxis, :] - self.s * shift_vector
        Z = np.concatenate([z_plus, z_minus, mu_verif[:, np.newaxis, :]], axis=1)
        if self.weights:
            z_weights = np.array(
                [1 / (np.exp(self.s**2 * 0.5) + 2 * self.K)] * (2 * self.K)
                + [1 / (1 + 2 * self.K * np.exp(-self.s**2 * 0.5))]
            )
        else:
            z_weights = np.ones(2 * self.K + 1) / self.K
        return Z, z_weights
