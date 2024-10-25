from pathlib import Path
import hydra
from hydra.utils import instantiate
from evaluation.face_recognition_test import Face_Fecognition_test
from evaluation.evaluate import instantiate_list, init_methods
from shutil import copyfile, rmtree
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def copy_template_images(
    data_path, test_template_path, template_id, image_paths, template_ids
):
    test_template_path.mkdir(exist_ok=True, parents=True)
    # copy template images
    for image_name in image_paths[template_ids == template_id]:
        in_image_name = data_path / "loose_crop" / image_name
        out_image_name = test_template_path / image_name
        copyfile(in_image_name, out_image_name)


def plot_unc_hist(
    work_dir,
    unc_name,
    unc_score,
    is_seen,
    false_ident,
    false_accept,
    false_reject,
):
    sns.set_theme()
    plt.figure(figsize=(12, 8))
    log_scale = True if unc_name == "vMF_unc" else False
    right_unc = np.concatenate(
        [
            unc_score[~is_seen][false_accept == False],
            unc_score[is_seen][
                np.logical_and(false_ident == False, false_reject == False)
            ],
        ]
    )
    false_accept_unc = unc_score[~is_seen][false_accept]
    false_ident_unc = unc_score[is_seen][
        np.logical_and(false_ident, false_reject == False)
    ]
    false_reject_unc = unc_score[is_seen][
        np.logical_and(false_ident == False, false_reject)
    ]
    false_ident_reject_unc = unc_score[is_seen][
        np.logical_and(false_ident, false_reject)
    ]
    unc = np.concatenate(
        [
            false_accept_unc,
            false_ident_unc,
            false_reject_unc,
            false_ident_reject_unc,
            right_unc,
        ]
    )
    unc += 1e-29
    error_kind = (
        ["false accept"] * len(false_accept_unc)
        + ["false ident"] * len(false_ident_unc)
        + ["false reject"] * len(false_reject_unc)
        + ["false reject/ident"] * len(false_ident_reject_unc)
        + ["right pred"] * len(right_unc)
    )
    data = pd.DataFrame({"unc": list(unc), "Error Kind": error_kind})
    sns.displot(
        data,
        kind="kde",
        x="unc",
        hue="Error Kind",
        log_scale=log_scale,
        common_norm=False,
    )
    plt.xlabel(f"{unc_name} score")
    plt.savefig(work_dir / f"{unc_name}_score_distr.png", dpi=300)


@hydra.main(
    config_path=str(Path(__file__).resolve().parents[1] / "configs/utils"),
    config_name=Path(__file__).stem,
    version_base="1.2",
)
def main(cfg):
    methods, method_task_type = init_methods(cfg)
    tasks_names = list(set(method_task_type))

    # instantiate datasets
    test_datasets = instantiate_list(cfg.test_datasets)
    dataset_names = [test_dataset.dataset_name for test_dataset in test_datasets]

    # instantiate method
    template_pooling = instantiate(methods[0].template_pooling_strategy)
    sampler = instantiate(methods[0].sampler)
    distance_function = instantiate(methods[0].distance_function)

    for test_dataset in test_datasets:
        dataset_name = test_dataset.dataset_name
        data_path = Path(test_dataset.dataset_path)
        media_list_path = (
            data_path / "meta" / f"{dataset_name.lower()}_face_tid_mid.txt"
        )
        pair_list_path = (
            data_path / "meta" / f"{dataset_name.lower()}_template_pair_label.txt"
        )
        img_path = data_path / "loose_crop"
        img_list_path = (
            data_path / "meta" / f"{dataset_name.lower()}_name_5pts_score.txt"
        )

        meta = pd.read_csv(media_list_path, sep=" ", skiprows=0, header=None).values
        image_paths = meta[:, 0]
        template_ids = meta[:, 1]

        embeddings_path = (
            Path(test_dataset.dataset_path)
            / f"embeddings/{methods[0].embeddings}_embs_{dataset_name}.npz"
        )
        tt = Face_Fecognition_test(
            task_type=tasks_names[0],
            method_name="test",
            recognition_method=None,
            sampler=sampler,
            distance_function=distance_function,
            test_dataset=test_dataset,
            embeddings_path=embeddings_path,
            template_pooling_strategy=template_pooling,
            use_detector_score=methods[0].use_detector_score,
            use_two_galleries=cfg.use_two_galleries,
            recompute_template_pooling=cfg.recompute_template_pooling,
            recognition_metrics=None,
            uncertainty_metrics=None,
        )

        used_galleries = ["g1", "g2"]
        galleries_data = [
            tt.get_template_subsets(
                getattr(tt.test_dataset, f"{g}_templates"),
                getattr(tt.test_dataset, f"{g}_ids"),
            )
            for g in used_galleries
        ]
        (
            probe_templates_feature,
            probe_template_unc,
            probe_unique_ids,
        ) = tt.get_template_subsets(
            tt.test_dataset.probe_templates, tt.test_dataset.probe_ids
        )
        probe_unique_templates = np.unique(
            tt.test_dataset.probe_templates, return_index=False
        )

        # sample probe feature vectors
        probe_templates_feature = tt.sampler(
            probe_templates_feature,
            probe_template_unc,
        )
        for gallery_name, (g_templates_feature, g_template_unc, g_unique_ids) in zip(
            used_galleries, galleries_data
        ):
            similarity = tt.distance_function(
                probe_templates_feature,
                probe_template_unc,
                g_templates_feature,
                g_template_unc,
            )

            g_unique_templates = np.unique(
                getattr(tt.test_dataset, f"{gallery_name}_templates"),
                return_index=False,
            )
            predictions = {}
            for method in methods:
                recognition_method = instantiate(method.recognition_method)
                # setup osr method and predict
                recognition_method.setup(similarity)
                predicted_id, was_rejected = recognition_method.predict()
                predicted_unc = recognition_method.predict_uncertainty(
                    probe_template_unc
                )
                predictions[method.pretty_name] = (
                    predicted_id,
                    was_rejected,
                    predicted_unc,
                )

            out_image_dir = Path(cfg.exp_dir) / dataset_name / gallery_name
            scf_unc_score = predictions["SCF"][2]
            vMF_unc_score = predictions["vMF"][2]

            is_seen = np.isin(probe_unique_ids, g_unique_ids)

            similar_gallery_class = g_unique_ids[predicted_id[is_seen]]
            false_ident = probe_unique_ids[is_seen] != similar_gallery_class
            false_ident_ids = probe_unique_templates[is_seen][false_ident]
            false_accept = was_rejected[~is_seen] == False
            false_accept_ids = probe_unique_templates[~is_seen][false_accept]
            false_reject = was_rejected[is_seen]
            false_reject_ids = probe_unique_templates[is_seen][false_reject]
            similarity = similarity[:, 0, :]

            hist_plot_path = out_image_dir / "score_hist"
            hist_plot_path.mkdir(exist_ok=True, parents=True)

            plot_unc_hist(
                hist_plot_path,
                "vMF_unc",
                vMF_unc_score,
                is_seen,
                false_ident,
                false_accept,
                false_reject,
            )
            plot_unc_hist(
                hist_plot_path,
                "scf_unc",
                scf_unc_score,
                is_seen,
                false_ident,
                false_accept,
                false_reject,
            )

            num_similar = 6
            for i in range(len(vMF_unc_score)):
                probe_unique_id = probe_unique_ids[i]
                vMF_unc = vMF_unc_score[i]
                scf_unc = scf_unc_score[i]
                probe_template_id = probe_unique_templates[i]
                if (
                    probe_template_id in false_ident_ids
                    and probe_template_id in false_reject_ids
                ):
                    case_name = "false_ident-false_reject"
                elif probe_template_id in false_accept_ids:
                    case_name = "false_accept"
                elif probe_template_id in false_reject_ids:
                    case_name = "false_reject"
                elif probe_template_id in false_ident_ids:
                    case_name = "false_ident"
                else:
                    continue
                most_similar_gallery_ids = np.argsort(similarity[i, :])[::-1]
                most_similar_templates = g_unique_templates[
                    most_similar_gallery_ids[:num_similar]
                ]
                probe_template_path = (
                    out_image_dir
                    / case_name
                    / f"scf-unc-{scf_unc}_vMF-unc-{vMF_unc}-probe_id-{str(probe_unique_id)}_teplate-id-{str(probe_template_id)}"
                )
                if probe_template_path.is_dir():
                    continue
                copy_template_images(
                    data_path,
                    probe_template_path
                    / f"probe_images_id-{str(probe_unique_id)}_template-{str(probe_template_id)}",
                    probe_template_id,
                    image_paths,
                    template_ids,
                )
                for j, template_id in enumerate(most_similar_templates):
                    cos_sim = similarity[i, most_similar_gallery_ids[j]]
                    id = g_unique_ids[most_similar_gallery_ids[j]]
                    test_template_path = (
                        probe_template_path
                        / f"close-gallery-id-{str(id)}_sim-{cos_sim}"
                    )
                    test_template_path.mkdir(exist_ok=True, parents=True)
                    copy_template_images(
                        data_path,
                        test_template_path,
                        template_id,
                        image_paths,
                        template_ids,
                    )


if __name__ == "__main__":
    main()
