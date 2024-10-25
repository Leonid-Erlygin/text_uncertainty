from scipy.stats import multivariate_normal
import numpy as np
from scipy.special import softmax
import numpy as np
from scipy.special import softmax
from scipy.stats import multivariate_normal


def compute_probalities(
    tester,
    sampler,
    likelihood_function,
    use_mean_z_estimate,
    num_z_samples,
    apply_softmax,
):
    """
    Computes probability for belonging to each class for each query image

    result of procedure: matrix $P \in \mathbb{R}^{n\times K}$, where n - number of query templates
    K - number of classes.
    Elements are probabilities $p_{ij} = \mathbb{P}(class_{i} = j)$ that i-th query template belongs to j-th class
    """

    z = []  # n x num_z_samples x 512
    sigma = []  # K x 512
    mu = []  # K x 512
    sigma_verif = []
    mu_verif = []

    for verif_template in tester.verification_templates():
        sigma_verif.append(verif_template.sigma_sq)
        mu_verif.append(verif_template.mu)
    sigma_verif = np.array(sigma_verif)
    mu_verif = np.array(mu_verif)
    z, z_weights = sampler(mu_verif, sigma_verif)
    z = np.array(z)

    for enroll_template in tester.enroll_templates():
        sigma.append(enroll_template.sigma_sq)
        mu.append(enroll_template.mu)
    sigma = np.array(sigma)
    mu = np.array(mu)

    a_ilj_final = likelihood_function(mu, sigma, z)

    if apply_softmax:
        p_ilj = softmax(
            a_ilj_final,
            axis=2,
        )
    else:
        sum_normalizer = np.sum(a_ilj_final, axis=2, keepdims=True)
        p_ilj = a_ilj_final / sum_normalizer

    p_ij = np.sum(
        p_ilj * z_weights[np.newaxis, :, np.newaxis],
        axis=1,
    )
    return p_ij
