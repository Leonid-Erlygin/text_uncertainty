import numpy as np
from tqdm import tqdm


def get_template_subsets(
    all_image_emb: np.ndarray,
    all_image_unc: np.ndarray,
    all_templates: np.ndarray,
    all_medias: np.ndarray,
    subject_ids: np.ndarray,
    choose_templates: np.ndarray,
):
    """
    selects features, uncertainty and medias of templates specified in choose_templates
    """
    assert subject_ids.shape[0] == choose_templates.shape[0]
    choose_templates_sort_id = np.argsort(
        choose_templates
    )  # is not stable sorting algorithm
    choose_templates_sorted = choose_templates[choose_templates_sort_id]
    subject_ids_sorted = subject_ids[choose_templates_sort_id]
    unique_templates, indices = np.unique(choose_templates_sorted, return_index=True)
    unique_subject_ids = subject_ids_sorted[indices]

    templates_emb_subset = []
    template_uncertainty_subset = []
    medias_subset = []
    for uqt in tqdm(unique_templates):
        ind_t = all_templates == uqt
        templates_emb_subset.append(all_image_emb[ind_t])
        template_uncertainty_subset.append(all_image_unc[ind_t])
        medias_subset.append(all_medias[ind_t])
    templates_emb_subset = np.concatenate(templates_emb_subset, axis=0)
    template_uncertainty_subset = np.concatenate(template_uncertainty_subset, axis=0)
    medias_subset = np.concatenate(medias_subset, axis=0)

    return (
        templates_emb_subset,
        template_uncertainty_subset,
        medias_subset,
        choose_templates_sorted,
        unique_subject_ids,
    )
