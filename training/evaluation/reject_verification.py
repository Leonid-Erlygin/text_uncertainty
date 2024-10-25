import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import sys

path = str(Path(__file__).parent.parent.parent.abspath())
sys.path.insert(1, path)

import training.utils.metrics as metrics
import training.evaluation.plots as plots


def get_rejected_tar_far(
    mu_1,
    mu_2,
    sigma_sq_1,
    sigma_sq_2,
    label_vec,
    distance_func,
    pair_uncertainty_func,
    FARs,
    uncertainty_mode="uncertainty",
    distance_ax=None,
    uncertainty_ax=None,
    rejected_portions=None,
    equal_uncertainty_enroll=False,
):
    # If something's broken, uncomment the line below

    # score_vec = force_compare(distance_func)(mu_1, mu_2, sigma_sq_1, sigma_sq_2)
    score_vec = distance_func(mu_1, mu_2, sigma_sq_1, sigma_sq_2)

    # if equal_uncertainty_enroll:
    #     sigma_sq_1 = np.ones_like(sigma_sq_1)

    uncertainty_vec = pair_uncertainty_func(mu_1, mu_2, sigma_sq_1, sigma_sq_2)
    if isinstance(uncertainty_vec, tuple):
        uncertainty_vec, positive = uncertainty_vec
    else:
        positive = None

    result_table = defaultdict(list)
    result_fars = defaultdict(list)

    if positive is None:
        sorted_indices = uncertainty_vec.argsort()
        score_vec = score_vec[sorted_indices]
        label_vec = label_vec[sorted_indices]
        uncertainty_vec = uncertainty_vec[sorted_indices]
        assert score_vec.shape == label_vec.shape

        if uncertainty_mode == "uncertainty":
            pass
        elif uncertainty_mode == "confidence":
            score_vec, label_vec, uncertainty_vec = (
                score_vec[::-1],
                label_vec[::-1],
                uncertainty_vec[::-1],
            )
        else:
            raise RuntimeError("Don't know this type uncertainty mode")

        for rejected_portion in tqdm(rejected_portions):
            cur_len = int(score_vec.shape[0] * (1 - rejected_portion))
            tars, fars, thresholds = metrics.ROC(
                score_vec[:cur_len], label_vec[:cur_len], FARs=FARs
            )
            for far, tar in zip(FARs, tars):
                result_table[far].append(tar)
            for wanted_far, real_far in zip(FARs, fars):
                result_fars[wanted_far].append(real_far)
    else:
        negative = np.invert(positive)
        sorted_indices_positive = uncertainty_vec[positive].argsort()
        sorted_indices_negative = uncertainty_vec[negative].argsort()
        print(
            f"Using separate thresholds for {len(sorted_indices_negative)} negative and {len(sorted_indices_positive)} positive pair"
        )

        if uncertainty_mode == "uncertainty":
            pass
        elif uncertainty_mode == "confidence":
            sorted_indices_positive = sorted_indices_positive[::-1]
            sorted_indices_negative = sorted_indices_negative[::-1]
        else:
            raise RuntimeError("Don't know this type uncertainty mode")

        uncertainty_vec_positive = uncertainty_vec[positive][sorted_indices_positive]
        score_vec_positive = score_vec[positive][sorted_indices_positive]
        label_vec_positive = label_vec[positive][sorted_indices_positive]

        uncertainty_vec_negative = uncertainty_vec[negative][sorted_indices_negative]
        score_vec_negative = score_vec[negative][sorted_indices_negative]
        label_vec_negative = label_vec[negative][sorted_indices_negative]

        beta_max = 0.05
        for rejected_portion in tqdm(rejected_portions):
            if rejected_portion < beta_max:
                beta = rejected_portion
            alpha = rejected_portion - (beta - rejected_portion) * (
                len(sorted_indices_positive) / len(sorted_indices_negative)
            )

            cur_len_positive = int(score_vec_positive.shape[0] * (1 - beta))
            cur_len_negative = int(score_vec_negative.shape[0] * (1 - alpha))

            score_vec_slice = np.concatenate(
                [
                    score_vec_positive[:cur_len_positive],
                    score_vec_negative[:cur_len_negative],
                ]
            )
            label_vec_slice = np.concatenate(
                [
                    label_vec_positive[:cur_len_positive],
                    label_vec_negative[:cur_len_negative],
                ]
            )
            tars, fars, thresholds = metrics.ROC(
                score_vec_slice, label_vec_slice, FARs=FARs
            )
            for far, tar in zip(FARs, tars):
                result_table[far].append(tar)
            for wanted_far, real_far in zip(FARs, fars):
                result_fars[wanted_far].append(real_far)

    plots.plot_distribution(
        score_vec,
        label_vec,
        xlabel_name="Distances",
        ylabel_name="Amount",
        ax=distance_ax,
    )

    plots.plot_distribution(
        uncertainty_vec,
        label_vec,
        xlabel_name="Uncertainties",
        ylabel_name="Amount",
        ax=uncertainty_ax,
    )

    return result_table
