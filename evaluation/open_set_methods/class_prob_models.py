from typing import Any

from matplotlib import cm, ticker
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import numpy as np
from evaluation.samplers import VonMisesFisher
from scipy.optimize import fsolve, minimize
from scipy.special import ive, hyp0f1, loggamma
from evaluation.metrics import FrrFarIdent
from utils.golden_section import golden_selection_search
import importlib

# from evaluation.template_pooling_strategies import PoolingDefault
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
    ) -> None:
        self.probe_feats = probe_feats
        self.probe_unc_scaled = probe_unc_scaled
        self.gallery_feats = gallery_feats
        self.gallery_unc = gallery_unc
        self.predict_T = predict_T
        self.target_far = target_far
        self.is_seen = is_seen
        self.env = env

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
        print(f"Found kappa {np.round(kappa,4)} for far {far}")
        return -np.abs(far - self.target_far) / self.target_far


class NNcalibration:
    def __init__(
        self,
        hidden_size,
        lr,
        epochs,
        weight,
        weight_decay,
        scheduler_params,
        train_weight=True,
        normalize_kl_by_test=False,
        random_subset_size=None,
        log_dir=None,
    ):
        self.device = torch.device("cuda")
        self.perceptron = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.BatchNorm1d(hidden_size, affine=True),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size, affine=True),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size, affine=True),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0),
        )
        self.perceptron.to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.weight = weight
        self.weight_decay = weight_decay
        self.scheduler_params = scheduler_params
        self.random_subset_size = random_subset_size
        self.log_dir = log_dir
        self.normalize_kl_by_test = normalize_kl_by_test
        self.train_weight = train_weight

    def train_calibration_parameters(self, kl_1, kl_2, true_pred_label, save_name):
        X = torch.tensor(
            np.concatenate([kl_1[None, :], kl_2[None, :]], axis=0).T,
            dtype=torch.float32,
            device=self.device,
        )
        # save validation normalization parameters
        self.X_mean_val = torch.mean(X, dim=0)
        self.X_std_val = torch.std(X, dim=0)
        X_norm = (X - self.X_mean_val) / self.X_std_val
        y = torch.tensor(
            true_pred_label.astype("bool"), dtype=torch.float32, device=self.device
        )

        if self.weight is None:
            true_pred_ratio = y.sum() / y.shape[0]
            print(true_pred_ratio.item())
            self.weight = true_pred_ratio.item()
        # weight = torch.tensor(self.weight, device=self.device)
        if self.train_weight:
            weight = torch.nn.Parameter(
                torch.tensor(self.weight, device=self.device), requires_grad=True
            )
        else:
            weight = torch.tensor(self.weight, device=self.device)

        scheduler_params = {
            "scheduler": "OneCycleLR",
            "params": {
                "max_lr": self.scheduler_params.max_lr,
                "steps_per_epoch": self.scheduler_params.steps_per_epoch,
                "epochs": self.epochs,
                "div_factor": self.scheduler_params.div_factor,
                "final_div_factor": self.scheduler_params.final_div_factor,
            },
            "interval": "epoch",
            "frequency": 1,
        }

        # loss_fn = nn.BCELoss(weight=weights)
        loss_fn = nn.BCELoss(reduce=False)
        self.perceptron.train()

        if self.random_subset_size is not None:
            weights = torch.zeros(
                int(X_norm.shape[0] * self.random_subset_size), device=self.device
            )
        else:
            pass

        if self.train_weight:
            optimizer = torch.optim.Adam(
                [*self.perceptron.parameters()] + [weight],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                [*self.perceptron.parameters()],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        scheduler = getattr(
            importlib.import_module("torch.optim.lr_scheduler"),
            scheduler_params["scheduler"],
        )(optimizer, **scheduler_params["params"])

        true_index = y == 1.0
        for iter in range(self.epochs):
            self.perceptron.train()
            optimizer.zero_grad()
            # sample train ds subset
            if self.random_subset_size is not None:
                indices = torch.randperm(X_norm.shape[0])[
                    : int(X_norm.shape[0] * self.random_subset_size)
                ]
                X_norm_subset = X_norm[indices]
                y_subset = y[indices]
                weights[y_subset == 1.0] = 1 - weight
                weights[y_subset == 0.0] = weight
                pred = self.perceptron(X_norm_subset)
                loss = (loss_fn(pred, y_subset) * weights).mean()
            else:
                pred = self.perceptron(X_norm)
                loss_element_wise = loss_fn(pred, y)
                loss = loss_element_wise[true_index].mean() * (
                    1 - torch.sigmoid(weight)
                ) + loss_element_wise[~true_index].mean() * torch.sigmoid(weight)
                # loss = (loss_fn(pred, y)* weights).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()
            # print(
            #     f"Iteration {iter}, Loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}"
            # )

            self.perceptron.eval()
            pred_eval = self.perceptron(X_norm)
            accuracy = np.mean(
                (pred_eval.detach().cpu().numpy() > 0.5) == y.cpu().numpy()
            )
            print(
                f"Iteration {iter}, Loss: {loss.item()}, accuracy: {accuracy.item()}, lr: {optimizer.param_groups[0]['lr']}"
            )
            print(torch.sigmoid(weight).item())
        # draw probs
        self.draw_dencity_plot(X_norm.cpu(), y.cpu(), save_name)

    def apply_calibration_transform(self, kl_1, kl_2, y, save_name):
        X = torch.tensor(
            np.concatenate([kl_1[None, :], kl_2[None, :]], axis=0).T,
            dtype=torch.float32,
            device=self.device,
        )
        if self.normalize_kl_by_test:
            self.X_mean_test = torch.mean(X, dim=0)
            self.X_std_test = torch.std(X, dim=0)
            X_norm = (X - self.X_mean_test) / self.X_std_test
        else:
            X_norm = (X - self.X_mean_val) / self.X_std_val
        self.perceptron.eval()
        predictions_perceptron = self.perceptron(X_norm)
        self.draw_dencity_plot(
            X_norm.cpu(), torch.tensor(y, dtype=torch.float32), save_name
        )
        unc = -predictions_perceptron.detach().cpu().numpy()
        return unc

    def draw_dencity_plot(self, X_norm, y, image_name):
        size = 500
        kl_1 = torch.linspace(
            X_norm[:, 0].min(), X_norm[:, 0].max(), size, device=self.device
        )
        kl_2 = torch.linspace(
            X_norm[:, 1].min(), X_norm[:, 1].max(), size, device=self.device
        )
        grid_x, grid_y = np.meshgrid(
            kl_1.cpu().numpy(), kl_2.cpu().numpy(), indexing="ij"
        )
        product = torch.cartesian_prod(kl_1, kl_2)

        self.perceptron.eval()
        predict_prob = self.perceptron(product)
        z = np.reshape(predict_prob.detach().cpu().numpy(), (size, size)).T
        z_min, z_max = z.min(), z.max()

        fig, ax = plt.subplots()
        cs = ax.contourf(grid_x, grid_y, z, cmap=cm.PuBu_r, vmin=z_min, vmax=z_max)
        ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        cbar = fig.colorbar(cs)
        sns.scatterplot(
            data={
                "kl_1": X_norm[:, 0].numpy(),
                "kl_2": X_norm[:, 1].numpy(),
                "true_pred_label": y.numpy(),
            },
            x="kl_1",
            y="kl_2",
            hue="true_pred_label",
            s=10,
            alpha=0.5,
        )
        log_dir = Path(self.log_dir) / "calibration_images"
        log_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(log_dir / f"{image_name}.png")


class MonteCarloPredictiveProb:
    def __init__(
        self,
        M: int,
        gallery_prior: str,
        emb_unc_model: str,
        beta: float,
        far: float,
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
        if gallery_prior == "vMF" or emb_unc_model == "power":
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
        if calibration_set is None:
            return
        self.gallery_pooled_templates_calib, self.probe_pooled_templates_calib = (
            prepare_calibration_dataset(calibration_set, calibration_embs_name)
        )
        self.calibration_transform = calibration_transform

    def setup(
        self,
        probe_feats: np.ndarray,
        probe_unc: np.ndarray,
        gallery_feats: np.ndarray,
        gallery_unc: np.ndarray,
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

        if g_unique_ids is not None and self.gallery_kappa == None:
            # find kappa
            is_seen = np.isin(probe_unique_ids, g_unique_ids)
            kappa_low = 300
            kappa_high = 1000
            max_iter = 10
            eps = 0.001
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
        if self.gallery_pooled_templates_calib is not None:
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
            kappa_low = 300
            kappa_high = 1000
            max_iter = 10
            eps = 0.001
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
            # calibratation_set_kappa = 519.1576
            print(
                f"Found kappa_calib {np.round(calibratation_set_kappa,4)} for far {self.far}"
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
                true_pred_label,
                f"{self.far}_calibration-set",
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
                self.kl_1, self.kl_2, true_pred_label, f"{self.far}_test-set"
            )
            # unc = -(self.alpha * self.kl_1 + (1 - self.alpha) * self.kl_2)
            # unc = -self.kl_1
            # unc = -self.kl_2
        return unc

    def compute_mean_probs_and_kl(
        self,
        mean: np.array,
        kappa: np.array,
        gallery_means: torch.nn.Parameter,
        gallery_kappas: torch.nn.Parameter,
        T: torch.nn.Parameter,
    ) -> Any:
        if type(gallery_means) == np.ndarray:
            cuda = torch.device("cuda:0")
            gallery_means = torch.tensor(gallery_means, device=cuda)
            gallery_kappas = torch.tensor(gallery_kappas, device=cuda)
        self.K = gallery_means.shape[0]
        zs = torch.tensor(self.sampler(mean, kappa), device=gallery_means.device)
        d = torch.tensor([mean.shape[-1]], device=gallery_means.device)
        similarities = torch.matmul(zs, gallery_means.T)
        if self.gallery_prior == "power":
            log_m_c_power = (
                torch.special.gammaln(d - 1 + gallery_kappas)
                + torch.special.gammaln(d / 2 + gallery_kappas)
                + gallery_kappas * np.log(2)
                - torch.special.gammaln(d / 2)
                - torch.special.gammaln(d - 1 + 2 * gallery_kappas)
            )
            # m_c_power = torch.exp(log_m_c_power)
            log_uniform_dencity = (
                torch.special.gammaln(d / 2) - np.log(2) - (d / 2) * np.log(np.pi)
            )
            log_normalizer = log_m_c_power + log_uniform_dencity
        assert self.gallery_prior == "power"
        # compute log z prob
        p_c = ((1 - self.beta) / self.K) ** (1 / T)
        sim_to_power_log = torch.multiply(
            torch.log(torch.add(similarities, 1, out=similarities), out=similarities),
            (gallery_kappas[..., :, 0] * (1 / T)),
            out=similarities,
        )
        logit_add = torch.add(
            sim_to_power_log, log_m_c_power[..., :, 0] * (1 / T), out=similarities
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
        sim_to_power_log = torch.multiply(
            torch.log(torch.add(similarities, 1, out=similarities), out=similarities),
            (gallery_kappas[..., :, 0] * (1 / T)),
            out=similarities,
        )
        pz_c_log = torch.add(
            sim_to_power_log,
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
        # kl_1 = torch.sum(
        #     mean_gallery_probs * torch.log(mean_gallery_probs), axis=1
        # )

        # compute kl_2
        # 1. compute log p(z_i|x)
        if self.emb_unc_model == "vMF":
            # TODO: sample more zs
            d = d.item()
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
                kappa[:, np.newaxis],  # * (1 / T),
                out=similarities,
            )
            log_p_z_given_x = torch.add(
                log_normalizer[:, np.newaxis],  # * (1 / T),
                sim_mult_kappa,
                out=similarities,
            )
            kl_2 = (self.beta) ** (1 / T) * torch.mean(
                (log_p_z_given_x - log_z_prob) / (self.beta + logit_sum),
                axis=1,
            )
            # kl_2 = self.beta * torch.mean( # default scf
            #     (log_p_z_given_x),
            #     axis=1,
            # )
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
