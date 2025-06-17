from typing import Any


import torch

import numpy as np
from evaluation.samplers import VonMisesFisher
from scipy.optimize import fsolve, minimize
from scipy.special import ive, hyp0f1, loggamma
from evaluation.metrics import FrrFarIdent
from utils.golden_section import golden_selection_search


from evaluation.open_set_methods.calibration_utils import prepare_calibration_dataset
from pathlib import Path


class GalleryMeans(torch.nn.Module):
    def __init__(self, init_means, device):
        super(GalleryMeans, self).__init__()
        self.gallery_means = torch.nn.Parameter(
            torch.tensor(init_means, dtype=torch.float64, device=device)
        )


class GalleryParams(torch.nn.Module):
    def __init__(self, init_mean, init_kappa, init_T, train_T, device):
        super(GalleryParams, self).__init__()
        self.gallery_means = torch.nn.Parameter(
            torch.tensor(init_mean, dtype=torch.float64, device=device)
        )


class FarLossCalc:
    def __init__(
        self,
        probe_feats,
        probe_unc_scaled,
        gallery_feats,
        gallery_unc,
        predict_T,
        target_far,
        is_seen,
        env,
        verbose=False,
    ) -> None:
        self.probe_feats = probe_feats
        self.probe_unc_scaled = probe_unc_scaled
        self.gallery_feats = gallery_feats
        self.gallery_unc = gallery_unc
        self.predict_T = predict_T
        self.target_far = target_far
        self.is_seen = is_seen
        self.env = env
        self.verbose = verbose

    def __call__(self, kappa: float) -> float:
        gallery_unc_scaled = np.ones_like(self.gallery_unc) * kappa
        out = self.env.compute_mean_probs_and_kl(
            self.probe_feats,
            self.probe_unc_scaled,
            self.gallery_feats,
            gallery_unc_scaled,
            self.predict_T,
        )
        mean_probs, kl_1, kl_2 = [x.cpu().detach().numpy() for x in out]

        oog_prob = 1 - np.sum(mean_probs, axis=-1, keepdims=True)
        all_prob = np.concatenate([mean_probs, oog_prob], axis=-1)
        was_rejected = np.argmax(all_prob, axis=-1) == (all_prob.shape[-1] - 1)
        far = np.mean(was_rejected[~self.is_seen] == False)
        if self.verbose:
            print(f"Found kappa {np.round(kappa,4)} for far {far}")
        return -np.abs(far - self.target_far) / self.target_far


class MonteCarloPredictiveProb:
    def __init__(
        self,
        gallery_prior: str,
        emb_unc_model: str,
        beta: float,
        far: float,
        M: int = 0,
        calibration_set=None,
        calibration_embs_name=None,
        calibration_transform=None,
        gallery_kappa: float = None,
        kappa_scale: float = 1.0,
        kappa_input_scale: float = 1.0,
        predict_T: float = 1.0,
        pred_uncertainty_type: str = "entropy",
        alpha: float = 0.5,
        log_dir: str = None,
    ) -> None:
        """
        params:
        M -- number of MC samples
        kappa_scale -- gallery unc multiplier
        gallery_prior -- model for p(z|c)
        emb_unc_model -- form of p(z|x)
        """
        self.M = M
        self.gallery_kappa = gallery_kappa
        self.kappa_scale = kappa_scale
        self.kappa_input_scale = kappa_input_scale
        assert gallery_prior in ["power", "vMF"]
        if emb_unc_model == "power":
            raise NotImplementedError
        assert emb_unc_model in ["vMF", "power"]
        self.emb_unc_model = emb_unc_model
        if self.emb_unc_model == "vMF":
            self.sampler = VonMisesFisher(self.M)
        else:
            raise ValueError

        self.gallery_prior = gallery_prior
        self.far = far
        self.beta = beta
        self.predict_T = predict_T
        self.pred_uncertainty_type = pred_uncertainty_type
        assert self.pred_uncertainty_type in ["entropy", "max_prob"]
        self.alpha = alpha
        self.log_dir = log_dir
        self.calibration_set = calibration_set
        self.calibration_embs_name = calibration_embs_name
        self.calibration_transform = calibration_transform

    def setup(
        self,
        probe_feats: np.ndarray,
        probe_unc: np.ndarray,
        gallery_feats: np.ndarray,
        gallery_unc: np.ndarray,
        dataset_name: str,
        g_unique_ids: np.ndarray = None,
        probe_unique_ids: np.ndarray = None,
    ):
        probe_unc_scaled = probe_unc * self.kappa_input_scale
        dtype = np.float64
        probe_feats = probe_feats.astype(dtype)
        probe_unc = probe_unc.astype(dtype)
        gallery_feats = gallery_feats.astype(dtype)
        gallery_unc = gallery_unc.astype(dtype)
        self.g_unique_ids = g_unique_ids
        self.probe_unique_ids = probe_unique_ids
        self.dataset_name = dataset_name
        if g_unique_ids is not None and self.gallery_kappa == None:
            # find kappa
            is_seen = np.isin(probe_unique_ids, g_unique_ids)
            kappa_low = 300
            kappa_high = 10000
            max_iter = 15
            eps = 0.0005
            far_loss_func = FarLossCalc(
                probe_feats,
                probe_unc_scaled,
                gallery_feats,
                gallery_unc,
                self.predict_T,
                self.far,
                is_seen,
                self,
            )
            self.gallery_kappa = golden_selection_search(
                kappa_high, kappa_low, eps, max_iter, far_loss_func
            )
            print(f"Found kappa {np.round(self.gallery_kappa,4)} for far {self.far}")

        gallery_unc_scaled = np.ones_like(gallery_unc) * self.gallery_kappa

        out = self.compute_mean_probs_and_kl(
            probe_feats,
            probe_unc_scaled,
            gallery_feats,
            gallery_unc_scaled,
            self.predict_T,
        )
        self.mean_probs, self.kl_1, self.kl_2 = [x.cpu().detach().numpy() for x in out]

        # get calibration set kl
        if self.calibration_set is not None:
            self.gallery_pooled_templates_calib, self.probe_pooled_templates_calib = (
                prepare_calibration_dataset(
                    self.calibration_set, self.calibration_embs_name
                )
            )
            self.calibration_transform = self.calibration_transform
            self.data_uncertainty_calib = self.probe_pooled_templates_calib["g1"][
                "template_pooled_data_unc"
            ]
            self.g_unique_ids_calib = self.gallery_pooled_templates_calib["g1"][
                "template_subject_ids_sorted"
            ]
            self.probe_unique_ids_calib = self.probe_pooled_templates_calib["g1"][
                "template_subject_ids_sorted"
            ]

            is_seen_calib = np.isin(
                self.probe_unique_ids_calib, self.g_unique_ids_calib
            )
            probe_feats_calib = self.probe_pooled_templates_calib["g1"][
                "template_pooled_features"
            ]
            # probe_templates_feature,
            probe_unc_calib = self.probe_pooled_templates_calib["g1"][
                "template_pooled_data_unc"
            ]
            gallery_feats_calib = self.gallery_pooled_templates_calib["g1"][
                "template_pooled_features"
            ]
            gallery_unc_calib = self.gallery_pooled_templates_calib["g1"][
                "template_pooled_data_unc"
            ]
            kappa_low = 200
            kappa_high = 10000
            max_iter = 15
            eps = 0.0005
            far_loss_func_calib = FarLossCalc(
                probe_feats_calib,
                probe_unc_calib,
                gallery_feats_calib,
                gallery_unc_calib,
                self.predict_T,
                self.far,
                is_seen_calib,
                self,
            )
            calibratation_set_kappa = golden_selection_search(
                kappa_high, kappa_low, eps, max_iter, far_loss_func_calib
            )
            gallery_unc_scaled_calib = (
                np.ones_like(gallery_unc_calib) * calibratation_set_kappa
            )

            out_calib = self.compute_mean_probs_and_kl(
                probe_feats_calib,
                probe_unc_calib,
                gallery_feats_calib,
                gallery_unc_scaled_calib,
                self.predict_T,
            )
            self.mean_probs_calib, self.kl_1_calib, self.kl_2_calib = [
                x.cpu().detach().numpy() for x in out_calib
            ]
            # calibrate
            predict_id_calib = np.argmax(self.mean_probs_calib, axis=-1)
            oog_prob = 1 - np.sum(self.mean_probs_calib, axis=-1, keepdims=True)
            all_prob = np.concatenate([self.mean_probs_calib, oog_prob], axis=-1)
            was_rejected_calib = np.argmax(all_prob, axis=-1) == (
                all_prob.shape[-1] - 1
            )
            true_pred_label = np.zeros(self.probe_unique_ids_calib.shape[0])
            error_calc = FrrFarIdent()
            error_calc(
                predict_id_calib,
                was_rejected_calib,
                self.g_unique_ids_calib,
                self.probe_unique_ids_calib,
            )
            true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
            true_pred_label[~error_calc.is_seen] = error_calc.true_reject
            self.calibration_transform.train_calibration_parameters(
                self.kl_1_calib,
                self.kl_2_calib,
                error_calc,
                dataset_name=self.calibration_set.dataset_name,
                far=self.far,
            )

    def predict(self):
        predict_probs = self.mean_probs
        predict_id = np.argmax(predict_probs, axis=-1)

        oog_prob = 1 - np.sum(predict_probs, axis=-1, keepdims=True)
        all_prob = np.concatenate([predict_probs, oog_prob], axis=-1)
        was_rejected = np.argmax(all_prob, axis=-1) == (all_prob.shape[-1] - 1)
        if self.log_dir is not None:
            # log error indicators and unc
            true_pred_label = np.zeros(self.probe_unique_ids.shape[0])
            error_calc = FrrFarIdent()
            error_calc(
                predict_id,
                was_rejected,
                self.g_unique_ids,
                self.probe_unique_ids,
            )
            true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
            true_pred_label[~error_calc.is_seen] = error_calc.true_reject
            np.savez(
                Path(self.log_dir) / f"kl_and_target_{self.predict_T}_M={self.M}.npz",
                kl_1=self.kl_1,
                kl_2=self.kl_2,
                true_pred_label=true_pred_label,
            )
        return predict_id, was_rejected

    def predict_uncertainty(self):
        if self.pred_uncertainty_type == "entropy":
            predict_id = np.argmax(self.mean_probs, axis=-1)
            oog_prob = 1 - np.sum(self.mean_probs, axis=-1, keepdims=True)
            all_prob = np.concatenate([self.mean_probs, oog_prob], axis=-1)
            was_rejected = np.argmax(all_prob, axis=-1) == (all_prob.shape[-1] - 1)
            true_pred_label = np.zeros(self.probe_unique_ids.shape[0])
            error_calc = FrrFarIdent()
            error_calc(
                predict_id,
                was_rejected,
                self.g_unique_ids,
                self.probe_unique_ids,
            )
            true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
            true_pred_label[~error_calc.is_seen] = error_calc.true_reject

            unc = self.calibration_transform.apply_calibration_transform(
                self.kl_1,
                self.kl_2,
                error_calc,
                dataset_name=self.dataset_name,
                far=self.far,
            )
        return unc

    def compute_mean_probs_and_kl(
        self,
        mean: np.array,
        kappa: np.array,
        gallery_means: torch.nn.Parameter,
        gallery_kappas: torch.nn.Parameter,
        T: torch.nn.Parameter,
    ) -> Any:
        gallery_kappas_np = gallery_kappas  # .copy()
        if type(gallery_means) == np.ndarray:
            cuda = torch.device("cuda:0")
            gallery_means = torch.tensor(gallery_means, device=cuda)
            gallery_kappas = torch.tensor(gallery_kappas, device=cuda)
        self.K = gallery_means.shape[0]
        zs = torch.tensor(self.sampler(mean, kappa), device=gallery_means.device)
        d = torch.tensor([mean.shape[-1]], device=gallery_means.device)
        d_np = mean.shape[-1]
        similarities = torch.matmul(zs, gallery_means.T)
        log_uniform_dencity = (
            torch.special.gammaln(d / 2) - np.log(2) - (d / 2) * np.log(np.pi)
        )
        if self.gallery_prior == "power":
            log_m_c = (
                torch.special.gammaln(d - 1 + gallery_kappas)
                + torch.special.gammaln(d / 2 + gallery_kappas)
                + gallery_kappas * np.log(2)
                - torch.special.gammaln(d / 2)
                - torch.special.gammaln(d - 1 + 2 * gallery_kappas)
            )
            log_normalizer = log_m_c + log_uniform_dencity
            pz_c_no_norm_log = torch.multiply(
                torch.log(
                    torch.add(similarities, 1, out=similarities), out=similarities
                ),
                (gallery_kappas[..., :, 0] * (1 / T)),
                out=similarities,
            )
        elif self.gallery_prior == "vMF":
            log_iv = (
                np.log(ive(d_np / 2 - 1, gallery_kappas_np, dtype=np.float64))
                + gallery_kappas_np
            )
            log_m_c = -np.log(
                hyp0f1(d_np / 2, gallery_kappas_np**2 / 4, dtype=np.float64)
            )

            log_normalizer = (
                (d_np / 2 - 1) * np.log(gallery_kappas_np)
                - d_np / 2 * np.log(2 * np.pi)
                - log_iv
            )
            log_m_c = torch.tensor(log_m_c, device=cuda)
            log_normalizer = torch.tensor(log_normalizer, device=cuda)
            pz_c_no_norm_log = torch.multiply(
                similarities,
                (gallery_kappas[..., :, 0] * (1 / T)),
                out=similarities,
            )
        else:
            raise ValueError
        # assert self.gallery_prior == "power"
        # compute log z prob
        p_c = ((1 - self.beta) / self.K) ** (1 / T)

        logit_add = torch.add(
            pz_c_no_norm_log, log_m_c[..., :, 0] * (1 / T), out=similarities
        )
        logit_exp = torch.exp(logit_add, out=similarities)
        logit_sum = (
            torch.sum(
                logit_exp,
                dim=-1,
            )
            * p_c
        )
        log_z_prob = (1 / T) * log_uniform_dencity + torch.log(
            logit_sum + (self.beta) ** (1 / T)
        )

        # compute gallery classes log prob
        similarities = torch.matmul(zs, gallery_means.T, out=similarities)
        if self.gallery_prior == "power":
            pz_c_no_norm_log = torch.multiply(
                torch.log(
                    torch.add(similarities, 1, out=similarities), out=similarities
                ),
                (gallery_kappas[..., :, 0] * (1 / T)),
                out=similarities,
            )
        elif self.gallery_prior == "vMF":
            pz_c_no_norm_log = torch.multiply(
                similarities,
                (gallery_kappas[..., :, 0] * (1 / T)),
                out=similarities,
            )
        else:
            raise ValueError
        pz_c_log = torch.add(
            pz_c_no_norm_log,
            (1 / T) * log_normalizer[..., :, 0],
            out=similarities,
        )

        gallery_log_probs = torch.sub(
            torch.add(
                pz_c_log, (1 / T) * np.log((1 - self.beta) / self.K), out=similarities
            ),
            log_z_prob[..., np.newaxis],
            out=similarities,
        )
        gallery_probs = torch.exp(gallery_log_probs, out=similarities)
        mean_gallery_probs = torch.mean(gallery_probs, axis=1)

        # compute kl_1
        kl_1 = torch.sum(
            mean_gallery_probs * torch.log(mean_gallery_probs / p_c), axis=1
        )

        # compute kl_2
        if self.emb_unc_model == "vMF":
            # TODO: sample more zs
            d = d.item()
            # p(z|x) dencity
            log_iv = np.log(ive(d / 2 - 1, kappa[:, 0], dtype=np.float64)) + kappa[:, 0]
            log_normalizer = (
                (d / 2 - 1) * np.log(kappa[:, 0]) - (d / 2) * np.log(2 * np.pi) - log_iv
            )

            mean = torch.tensor(mean, device=cuda)
            kappa = torch.tensor(kappa[:, 0], device=cuda)
            log_normalizer = torch.tensor(log_normalizer, device=cuda)

            similarities = torch.sum(zs * mean[:, np.newaxis, :], axis=2)

            sim_mult_kappa = torch.multiply(
                similarities,
                kappa[:, np.newaxis],
                out=similarities,
            )
            log_p_z_given_x = torch.add(
                log_normalizer[:, np.newaxis],
                sim_mult_kappa,
                out=similarities,
            )
            kl_2 = (self.beta) ** (1 / T) * torch.mean(
                (
                    (1 / T - 1) * (np.log(self.beta) + log_uniform_dencity)
                    + log_p_z_given_x
                    - log_z_prob
                )
                / (self.beta ** (1 / T) + logit_sum),
                axis=1,
            )
        else:
            raise ValueError
        return mean_gallery_probs, kl_1, kl_2


# class MonteCarloPredictiveProbV2:
#     def __init__(
#         self,
#         M: int,
#         gallery_prior: str,
#         emb_unc_model: str,
#         beta: float,
#         far: float,
#         gallery_kappa: float = None,
#         ood_kappa: float = None,
#         kappa_scale: float = 1.0,
#         kappa_input_scale: float = 1.0,
#         predict_T: float = 1.0,
#         pred_uncertainty_type: str = "entropy",
#     ) -> None:
#         """
#         Here we assume several out-of-gallery class centers in order to enhance false reject recognition rate

#         params:
#         M -- number of MC samples
#         kappa_scale -- gallery unc multiplier
#         gallery_prior -- model for p(z|c)
#         emb_unc_model -- form of p(z|x)
#         """
#         self.M = M
#         self.gallery_kappa = gallery_kappa
#         self.ood_kappa = ood_kappa
#         self.kappa_scale = kappa_scale
#         self.kappa_input_scale = kappa_input_scale
#         assert gallery_prior in ["power", "vMF"]
#         if gallery_prior == "vMF" or emb_unc_model == "power":
#             raise NotImplementedError
#         assert emb_unc_model in ["vMF", "power"]
#         if emb_unc_model == "vMF":
#             self.sampler = VonMisesFisher(self.M)

#         self.gallery_prior = gallery_prior
#         self.far = far
#         self.beta = beta
#         self.predict_T = predict_T
#         self.pred_uncertainty_type = pred_uncertainty_type
#         assert self.pred_uncertainty_type in ["entropy", "max_prob"]

#     def setup(
#         self,
#         probe_feats: np.ndarray,
#         probe_unc: np.ndarray,
#         gallery_feats: np.ndarray,
#         gallery_unc: np.ndarray,
#         g_unique_ids: np.ndarray = None,
#         probe_unique_ids: np.ndarray = None,
#     ):
#         probe_unc_scaled = probe_unc * self.kappa_input_scale
#         dtype = np.float64
#         probe_feats = probe_feats.astype(dtype)
#         probe_unc = probe_unc.astype(dtype)
#         gallery_feats = gallery_feats.astype(dtype)
#         gallery_unc = gallery_unc.astype(dtype)
#         self.oog_classes_number = probe_feats.shape[-1] * 2
#         gallery_unc_scaled = np.concatenate(
#             [
#                 np.ones_like(gallery_unc) * self.gallery_kappa,
#                 np.ones((self.oog_classes_number, 1)) * self.ood_kappa,
#             ]
#         )
#         self.mean_probs = (
#             self.compute_mean_probs(
#                 probe_feats,
#                 probe_unc_scaled,
#                 gallery_feats,
#                 gallery_unc_scaled,
#                 self.predict_T,
#             )
#             .cpu()
#             .detach()
#             .numpy()
#         )
#         if self.M != 0:
#             # self.mean_probs_pred = None

#             self.sampler = VonMisesFisher(0)
#             self.beta = 0.99
#             # self.beta = 0.594
#             # gallery_unc_scaled = np.ones_like(gallery_unc) * self.gallery_kappa
#             self.mean_probs_pred = (
#                 self.compute_mean_probs(
#                     probe_feats,
#                     probe_unc_scaled,
#                     gallery_feats,
#                     gallery_unc_scaled,
#                     4,
#                 )
#                 .cpu()
#                 .detach()
#                 .numpy()
#             )
#         else:
#             self.mean_probs_pred = None

#     def predict(self):
#         if self.mean_probs_pred is not None:
#             predict_probs = self.mean_probs_pred
#         else:
#             predict_probs = self.mean_probs
#         predict_id = np.argmax(predict_probs[:, : -self.oog_classes_number], axis=-1)
#         return predict_id, np.argmax(predict_probs, axis=-1) >= (
#             predict_probs.shape[-1] - self.oog_classes_number
#         )

#     def predict_uncertainty(self):
#         # TODO: separate epistemic and aleatoric uncertainties
#         if self.pred_uncertainty_type == "entropy":
#             unc = -np.sum(self.mean_probs * np.log(self.mean_probs), axis=-1)
#         elif self.pred_uncertainty_type == "max_prob":
#             unc = -np.max(self.mean_probs, axis=-1)
#         return unc

#     def compute_mean_probs(
#         self,
#         mean: np.array,
#         kappa: np.array,
#         gallery_means: torch.nn.Parameter,
#         class_kappas: torch.nn.Parameter,
#         T: torch.nn.Parameter,
#     ) -> Any:
#         if type(gallery_means) == np.ndarray:
#             # inference
#             cuda = torch.device("cuda:0")
#             gallery_means = torch.tensor(gallery_means, device=cuda)
#             class_kappas = torch.tensor(class_kappas, device=cuda)
#         # add out-of-gallery kappas
#         self.K = gallery_means.shape[0]
#         L = self.oog_classes_number
#         zs = torch.tensor(self.sampler(mean, kappa), device=gallery_means.device)
#         d = torch.tensor([mean.shape[-1]], device=gallery_means.device)
#         similarities = torch.matmul(zs, gallery_means.T)
#         similarities = torch.cat([similarities, zs, -zs], dim=-1)
#         # compute log z prob
#         p_c = ((1 - self.beta) / self.K) ** (1 / T)
#         p_out = (self.beta / L) ** (1 / T)
#         sim_to_power = torch.pow(
#             torch.add(similarities, 1, out=similarities),
#             (class_kappas[..., :, 0] * (1 / T)),
#             out=similarities,
#         )
#         kappa_g = class_kappas[0, 0]
#         kappa_o = class_kappas[-1, 0]
#         alpha_galley_over_alpha_out = torch.exp(
#             (1 / T)
#             * (
#                 (kappa_g - kappa_o) * np.log(2.0)
#                 + torch.special.gammaln(d - 1 + kappa_o)
#                 + torch.special.gammaln((d - 1) / 2 + kappa_g)
#                 - torch.special.gammaln(d - 1 + kappa_g)
#                 - torch.special.gammaln((d - 1) / 2 + kappa_o)
#             )
#         )
#         prior_class_probs = torch.tensor(
#             [p_c] * self.K + [p_out * alpha_galley_over_alpha_out] * L, device=cuda
#         )
#         class_likelihoods = torch.multiply(
#             sim_to_power, prior_class_probs[..., :], out=similarities
#         )
#         z_prob = torch.sum(class_likelihoods, dim=-1)
#         gallery_probs = torch.divide(
#             class_likelihoods,
#             z_prob[..., np.newaxis],
#             out=similarities,
#         )
#         mean_gallery_probs = torch.mean(gallery_probs, axis=1)
#         return mean_gallery_probs
