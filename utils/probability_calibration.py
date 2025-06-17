import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import hydra
from hydra.utils import instantiate
from evaluation.recognition_test import Recognition_test
from evaluation.evaluate import instantiate_list, init_methods
from shutil import copyfile, rmtree
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.reliability_diagrams import reliability_diagram, compute_calibration
import torch
from scipy.optimize import fsolve, minimize
from itertools import product


def train_T_scipy(
    cfg,
    recognition_method,
    true_id,
    is_seen,
    sim_tensor,
    probe_unique_ids,
    g_unique_ids,
    tau,
):
    true_id_class_index = np.zeros_like(true_id)
    true_id_class_index[~is_seen] = sim_tensor.shape[2]

    true_id_class_index[is_seen] = np.argmax(
        probe_unique_ids[is_seen, np.newaxis] == g_unique_ids[np.newaxis, :],
        axis=1,
    )
    true_id_class_index_tensor = torch.tensor(true_id_class_index, dtype=torch.long)

    res = minimize(
        recognition_method.posterior_prob.compute_nll,
        1,
        (sim_tensor, true_id_class_index_tensor),
    )
    print(f"Optimal T: {res.x} for tau: {tau}")
    return res.x[0]


def train_T(
    cfg,
    recognition_method,
    true_id,
    is_seen,
    sim_tensor,
    probe_unique_ids,
    g_unique_ids,
):
    true_id_class_index = np.zeros_like(true_id)
    true_id_class_index[~is_seen] = sim_tensor.shape[2]

    true_id_class_index[is_seen] = np.argmax(
        probe_unique_ids[is_seen, np.newaxis] == g_unique_ids[np.newaxis, :],
        axis=1,
    )
    T = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    optimizer = torch.optim.SGD([T], lr=cfg.lr, momentum=0.5)

    true_id_class_index_tensor = torch.tensor(true_id_class_index, dtype=torch.long)
    for iter in range(cfg.iter_num):
        optimizer.zero_grad()

        loss = recognition_method.posterior_prob.compute_nll(
            T, sim_tensor, true_id_class_index_tensor
        )
        loss.backward()
        optimizer.step()
        print(
            f"Iteration {iter}, Loss: {loss.item()}, T: {T.item()}, T_grad: {T.grad.item()}"
        )


def compute_ece(T, conf_id, true_labels, pred_labels, num_bins):
    min_kappa = 400
    max_kappa = 2000
    data_uncertainty_norm = (conf_id - min_kappa) / (max_kappa - min_kappa)
    confidences = (data_uncertainty_norm) ** (1 / T)
    return compute_calibration(true_labels, pred_labels, confidences, num_bins)[
        "expected_calibration_error"
    ]


def train_T_ece(cfg, conf_id, true_labels, pred_labels):
    res = minimize(compute_ece, 1, (conf_id, true_labels, pred_labels, cfg.num_bins))
    print(res.x)
    return res.x[0]


@hydra.main(
    config_path=str(Path(__file__).resolve().parents[1] / "configs/utils"),
    config_name=Path(__file__).stem,
    version_base="1.2",
)
def main(cfg):
    methods, method_task_type = init_methods(cfg)
    tasks_names = list(set(method_task_type))
    # assert len(methods) == 1
    # instantiate datasets
    test_datasets = instantiate_list(cfg.test_datasets)
    dataset_names = [test_dataset.dataset_name for test_dataset in test_datasets]

    # instantiate method
    gallery_template_pooling_strategy = instantiate(
        methods[0].gallery_template_pooling_strategy
    )
    probe_template_pooling_strategy = instantiate(
        methods[0].probe_template_pooling_strategy
    )
    sampler = instantiate(methods[0].sampler)
    distance_function = instantiate(methods[0].distance_function)

    for test_dataset in test_datasets:
        dataset_name = test_dataset.dataset_name
        data_path = Path(test_dataset.dataset_path)
        embeddings_path = (
            Path(test_dataset.dataset_path)
            / f"embeddings/{methods[0].embeddings}_embs_{dataset_name}.npz"
        )
        tt = Recognition_test(
            task_type=tasks_names[0],
            method_name="test",
            recognition_method=None,
            sampler=sampler,
            distance_function=distance_function,
            test_dataset=test_dataset,
            embeddings_path=embeddings_path,
            gallery_template_pooling_strategy=gallery_template_pooling_strategy,
            probe_template_pooling_strategy=probe_template_pooling_strategy,
            use_detector_score=methods[0].use_detector_score,
            use_two_galleries=cfg.use_two_galleries,
            recompute_template_pooling=cfg.recompute_template_pooling,
            recognition_metrics=None,
            uncertainty_metrics=None,
        )

        used_galleries = ["g1"]
        if cfg.use_two_galleries:
            used_galleries += ["g2"]

        for gallery_name in used_galleries:
            # sample probe feature vectors
            probe_templates_feature = tt.sampler(
                tt.probe_pooled_templates[gallery_name]["template_pooled_features"],
                tt.probe_pooled_templates[gallery_name]["template_pooled_data_unc"],
            )
            similarity = tt.distance_function(
                probe_templates_feature,
                tt.probe_pooled_templates[gallery_name]["template_pooled_data_unc"],
                tt.gallery_pooled_templates[gallery_name]["template_pooled_features"],
                tt.gallery_pooled_templates[gallery_name]["template_pooled_data_unc"],
            )
            g_unique_ids = tt.gallery_pooled_templates[gallery_name][
                "template_subject_ids_sorted"
            ]
            probe_unique_ids = tt.probe_pooled_templates[gallery_name][
                "template_subject_ids_sorted"
            ]
            probe_template_unc = tt.probe_pooled_templates[gallery_name][
                "template_pooled_data_unc"
            ]

            true_id = np.zeros(similarity.shape[0])
            is_seen = np.isin(probe_unique_ids, g_unique_ids)
            true_id[is_seen] = probe_unique_ids[is_seen]
            true_id[~is_seen] = -1

            taus = np.linspace(cfg.tau_range[0], cfg.tau_range[1], cfg.tau_range[2])
            temps = []
            for method, tau in product(methods, taus):
                recognition_method = instantiate(method.recognition_method)
                recognition_method.kappa = tau
                gallery_ids_with_imposter_id = np.concatenate([g_unique_ids, [-1]])
                sim_tensor = torch.tensor(similarity)
                recognition_method.setup(similarity)

                if "SCF" in method.pretty_name or "Be" in method.pretty_name:
                    # here we use scf concentrations as best class prob estimate

                    predict_id, was_rejected = recognition_method.predict()
                    predict_id[was_rejected] = gallery_ids_with_imposter_id.shape[0] - 1
                    predict_id = gallery_ids_with_imposter_id[predict_id]
                    conf_id = -recognition_method.predict_uncertainty(
                        probe_template_unc
                    )
                    # assert recognition_method.T_data_unc == 1

                    if cfg.train_T:
                        T = train_T_ece(
                            cfg, probe_template_unc[:, 0], true_id, predict_id
                        )
                        if "SCF" in method.pretty_name:
                            recognition_method.T_data_unc = T
                        else:
                            recognition_method.T = T
                        conf_id = -recognition_method.predict_uncertainty(
                            probe_template_unc
                        )
                else:
                    T = recognition_method.T
                    if cfg.train_T:
                        trainer = train_T_scipy
                        T = trainer(
                            cfg,
                            recognition_method,
                            true_id,
                            is_seen,
                            sim_tensor,
                            probe_unique_ids,
                            g_unique_ids,
                            tau,
                        )

                    recognition_method.kappa = tau
                    recognition_method.T = T
                    class_log_probs = recognition_method.get_class_log_probs(similarity)

                    predict_id = gallery_ids_with_imposter_id[
                        np.argmax(class_log_probs, axis=-1)
                    ]
                    conf_id = np.exp(np.max(class_log_probs, axis=-1))
                temps.append(T)
                plt.style.use("seaborn")

                plt.rc("font", size=12)
                plt.rc("axes", labelsize=12)
                plt.rc("xtick", labelsize=12)
                plt.rc("ytick", labelsize=12)
                plt.rc("legend", fontsize=12)

                plt.rc("axes", titlesize=16)
                plt.rc("figure", titlesize=16)

                title = (
                    method.pretty_name + f"T:{np.round(T, 4)},tau:{np.round(tau, 2)}"
                )

                fig = reliability_diagram(
                    true_id,
                    predict_id,
                    conf_id,
                    num_bins=cfg.num_bins,
                    draw_ece=True,
                    draw_bin_importance=cfg.draw_bin_importance,
                    draw_averages=cfg.draw_averages,
                    title=title,
                    figsize=(6, 6),
                    dpi=300,
                    return_fig=True,
                )
                out_file = (
                    Path(cfg.exp_dir) / f"{dataset_name}_{gallery_name}_{title}.png"
                )
                fig.savefig(out_file, dpi=300)
                plt.close(fig)
            np.savez(
                Path(cfg.exp_dir) / f"{method.pretty_name}_learned_temp.npz",
                taus=taus,
                temps=temps,
            )
    print(cfg.exp_dir)


if __name__ == "__main__":
    main()
