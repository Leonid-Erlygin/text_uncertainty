# evaluation/distance_functions/open_set_identification.py

import numpy as np
from scipy.linalg import inv
from evaluation.distance_functions.open_set_identification.abc import Abstract1NEval


class MahalanobisDistance(Abstract1NEval):
    def __init__(self, reg_eps: float = 1e-6):
        self.reg_eps = reg_eps
        self.class_means = None  # dict: cls -> mean (D,)
        self.inv_covs = None  # dict: cls -> inv_cov (D, D)
        self.classes = None  # list of class IDs
        self.D = None  # feature dimension (e.g., 768)

    def _extract_labels_and_feats(self, gallery_feats_with_labels: np.ndarray):
        """
        gallery_feats_with_labels: (N, D+1), first column = class ID (float or int)
        Returns:
            labels: (N,)
            feats: (N, D)
        """
        labels = gallery_feats_with_labels[:-6, 0]
        feats = gallery_feats_with_labels[:-6, 1:]
        return labels, feats

    def setup_from_gallery(self, gallery_feats_with_labels: np.ndarray):
        """
        Called during recognition_method.setup() — but we'll do it lazily in __call__ if needed.
        However, to keep interface identical to CosineSim, we compute stats on first __call__.
        But better: compute once and cache.
        """
        labels, feats = self._extract_labels_and_feats(gallery_feats_with_labels)
        self.D = feats.shape[1]

        unique_classes = np.unique(labels)
        self.classes = unique_classes
        self.class_means = {}
        self.inv_covs = {}

        for cls in unique_classes:
            mask = labels == cls
            cls_feats = feats[mask]  # (N_cls, D)

            mean = cls_feats.mean(axis=0)
            cov = np.cov(cls_feats, rowvar=False)
            cov_reg = cov + self.reg_eps * np.eye(self.D)
            inv_cov = inv(cov_reg)

            self.class_means[cls] = mean
            self.inv_covs[cls] = inv_cov

    def __call__(
        self,
        probe_feats: np.ndarray,
        probe_unc: np.ndarray,
        gallery_feats_with_labels: np.ndarray,  # (N, D+1) — labels concatenated!
        gallery_unc: np.ndarray,
    ):
        """
        probe_feats: (n, num_z, D) — D = 768, NOT 769!
        gallery_feats_with_labels: (N, D+1) — first column = class ID
        Returns: (n, num_z, C) — similarity-like scores (higher = more similar)
        """
        # Lazy initialization: compute stats on first call
        if self.class_means is None:
            self.setup_from_gallery(gallery_feats_with_labels)

        n, num_z, D_probe = probe_feats.shape
        assert D_probe == self.D, f"Probe dim {D_probe} != gallery feat dim {self.D}"

        C = len(self.classes)
        scores = np.empty((n, num_z, C), dtype=np.float32)

        for i, cls in enumerate(self.classes):
            mean = self.class_means[cls]  # (D,)
            inv_cov = self.inv_covs[cls]  # (D, D)

            # Compute Mahalanobis distance from each probe sample to this class
            diff = probe_feats - mean  # (n, num_z, D)
            mid = diff @ inv_cov  # (n, num_z, D)
            dist_sq = np.einsum("nzd,nzd->nz", mid, diff)
            dist = np.sqrt(np.maximum(dist_sq, 0))
            scores[:, :, i] = -dist  # higher = more similar (like cosine)

        return scores
