import numpy as np

from typing import Any


class BernoulliVariance:
    def __call__(self, similarity: np.ndarray, probe_score: np.ndarray, tau) -> Any:
        s = probe_score
        conf_score = np.abs(s - tau)
        return -conf_score


class RandomScore:
    def __call__(self, similarity: np.ndarray, probe_score: np.ndarray, tau) -> Any:
        unc_score = np.arange(probe_score.shape[0])
        rng = np.random.default_rng(1)
        rng.shuffle(unc_score)
        return unc_score


class OracleScore:
    def __call__(self, similarity: np.ndarray, probe_score: np.ndarray, tau) -> Any:
        unc_score = np.arange(probe_score.shape[0])
        rng = np.random.default_rng(1)
        rng.shuffle(unc_score)
        return unc_score
