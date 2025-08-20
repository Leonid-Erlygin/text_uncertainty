import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import importlib
import numpy as np
from torch.utils.data import Dataset
from training.models.due import dkl
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.likelihoods import GaussianLikelihood


class DUESphereConfidenceFace(LightningModule):
    def __init__(
        self,
        target_feature_vector_model: torch.nn.Module,
        softmax_weights: torch.nn.Module,
        feature_extractor: torch.nn.Module,
        train_dataset: Dataset,
        n_inducing_points: int,
        kernel: str,
        optimizer_params,
        scheduler_params,
        permute_batch=False,
    ):
        super().__init__()
        self.target_feature_vector_model = target_feature_vector_model
        self.target_feature_vector_model.eval()
        self.softmax_weights = softmax_weights.softmax_weights
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.permute_batch = permute_batch
        self.validation_step_outputs = []

        initial_inducing_points, initial_lengthscale = dkl.initial_values(
            train_dataset, feature_extractor, n_inducing_points
        )

        gp = dkl.GP(
            num_outputs=1,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=kernel,
        )

        self.model = dkl.DKL(feature_extractor, gp)
        self.likelihood = GaussianLikelihood()
        elbo_fn = VariationalELBO(
            self.likelihood, self.model.gp, num_data=len(train_dataset)
        )
        self.loss_fn = lambda x, y: -elbo_fn(x, y)

    def forward(self, x):
        cosine_sim = self.model(x)
        backbone_outputs = self.target_feature_vector_model(x)
        return backbone_outputs["feature"], cosine_sim

    def training_step(self, batch):
        images, labels = batch
        feature, cosine_sim_pred = self(images)
        wc = self.softmax_weights[labels, :]
        cosine_sim_true = torch.sum(feature * wc, dim=1, keepdim=True)

        losses = self.loss_fn(cosine_sim_pred, cosine_sim_true)

        total_loss = losses.mean()

        self.log("train_loss", total_loss.item(), prog_bar=True)
        self.log("cos pred", cosine_sim_pred.mean.mean().item())
        self.log("cos true", cosine_sim_true.mean().item())

        return total_loss

    def configure_optimizers(self):

        params = [*self.model.parameters(), *self.likelihood.parameters()]
        optimizer = getattr(
            importlib.import_module(self.optimizer_params["optimizer_path"]),
            self.optimizer_params["optimizer_name"],
        )(
            params,
            **self.optimizer_params["params"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": getattr(
                    importlib.import_module("torch.optim.lr_scheduler"),
                    self.scheduler_params["scheduler"],
                )(optimizer, **self.scheduler_params["params"]),
                "interval": self.scheduler_params["interval"],
            },
        }

    def predict_step(self, batch, batch_idx):
        if len(batch) == 4:
            # five ds pred
            images_batch, _, _, _ = batch
        elif len(batch) == 2:
            # ms1m pred
            images_batch, labels = batch
            return self(images_batch)
        else:
            images_batch = batch
        if self.permute_batch:
            images_batch = images_batch.permute(0, 3, 1, 2)
        return self(images_batch)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        if self.permute_batch:
            images = images.permute(0, 3, 1, 2)
        features, cosine_sim_pred = self(images)
        self.validation_step_outputs.append([features, cosine_sim_pred.mean, labels])

    def on_validation_epoch_end(self):
        # here we use cosine sim as confidence measure
        image_input_feats = (
            torch.cat([batch[0] for batch in self.validation_step_outputs], axis=0)
            .cpu()
            .numpy()
        )
        log_kappa = (
            torch.cat([batch[1] for batch in self.validation_step_outputs], axis=0)
            .cpu()
            .numpy()
        )
        labels = (
            torch.cat([batch[2] for batch in self.validation_step_outputs], axis=0)
            .cpu()
            .numpy()
        )
        self.validation_step_outputs.clear()
        kappa = np.exp(log_kappa)

        unc_indexes = np.argsort(-kappa)
        fractions = [0, 0.5, 10]
        fractions_linspace = np.linspace(fractions[0], fractions[1], fractions[2])
        accuracies = []
        weights = F.normalize(self.softmax_weights, p=2, dim=1).detach().cpu().numpy()
        for fraction in fractions_linspace:
            good_idx = unc_indexes[: int((1 - fraction) * kappa.shape[0])]
            good_feat = image_input_feats[good_idx]
            good_labels = labels[good_idx]
            predictions = np.argmax(good_feat @ weights.T, axis=-1)
            accuracy = np.mean(predictions == good_labels)
            accuracies.append(accuracy)
        unc_auc_pr = np.mean(accuracies) * fractions[-2]

        # random
        unc_indexes = np.arange(kappa.shape[0])
        rng = np.random.default_rng(1)
        rng.shuffle(unc_indexes)
        accuracies_random = []
        for fraction in fractions_linspace:
            good_idx = unc_indexes[: int((1 - fraction) * kappa.shape[0])]
            good_feat = image_input_feats[good_idx]
            good_labels = labels[good_idx]
            predictions = np.argmax(good_feat @ weights.T, axis=-1)
            accuracy = np.mean(predictions == good_labels)
            accuracies_random.append(accuracy)
        random_auc_pr = np.mean(accuracies_random) * fractions[-2]

        # oracle

        unc_oracle = np.zeros(kappa.shape[0])
        predictions = np.argmax(image_input_feats @ weights.T, axis=-1)
        errors = predictions != labels
        unc_oracle[errors] = 1
        unc_indexes = np.argsort(unc_oracle)
        accuracies_oracle = []
        for fraction in fractions_linspace:
            good_idx = unc_indexes[: int((1 - fraction) * kappa.shape[0])]
            good_feat = image_input_feats[good_idx]
            good_labels = labels[good_idx]
            predictions = np.argmax(good_feat @ weights.T, axis=-1)
            accuracy = np.mean(predictions == good_labels)
            accuracies_oracle.append(accuracy)
        oracle_auc_pr = np.mean(accuracies_oracle) * fractions[-2]

        self.log("random auc", random_auc_pr)
        self.log("oracle auc", oracle_auc_pr)
        self.log("unc auc", unc_auc_pr)

        self.log("PPR", (unc_auc_pr - random_auc_pr) / (oracle_auc_pr - random_auc_pr))
