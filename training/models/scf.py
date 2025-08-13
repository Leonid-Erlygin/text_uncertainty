import torch
import torch.nn.functional as F
import math
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import BasePredictionWriter, Callback
import importlib
from pathlib import Path
import numpy as np
from evaluation.evaluate import instantiate_list
from evaluation.template_pooling_strategies import PoolingDefault


class Prediction_writer(BasePredictionWriter):
    def __init__(self, output_dir: str, file_name: str, write_interval: str):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.file_name = file_name

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        embs = torch.cat([batch[0] for batch in predictions], axis=0).numpy()
        unc = torch.cat([batch[1] for batch in predictions], axis=0).numpy()
        print(embs.shape, unc.shape)
        np.savez(self.output_dir / f"{self.file_name}.npz", embs=embs, unc=unc)


class SoftmaxWeights(torch.nn.Module):
    def __init__(
        self, softmax_weights_path: str, radius: int, requires_grad=False
    ) -> None:
        super().__init__()
        softmax_weights = torch.load(softmax_weights_path)
        softmax_weights_norm = torch.norm(
            softmax_weights, dim=1, keepdim=True
        )  # [N, 512]
        softmax_weights = (
            softmax_weights / softmax_weights_norm * radius
        )  # $ w_c \in rS^{d-1} $

        self.softmax_weights = torch.nn.Parameter(
            softmax_weights, requires_grad=requires_grad
        )


class SphereConfidenceFace(LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        scf_loss: torch.nn.Module,
        optimizer_params,
        scheduler_params,
        permute_batch: bool,
        softmax_weights: torch.nn.Module = None,
        validation_dataset=None,
        template_pooling_strategy=None,
        recognition_method=None,
        verification_metrics=None,
        verification_uncertainty_metrics=None,
        predict_kappa_by_input=False,
        backbone_input=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        self.head = head
        self.scf_loss = scf_loss
        # bad style code:
        if softmax_weights is None:
            # assume that weights are stored in the backbone as in case of whale dataset
            self.softmax_weights = self.backbone.backbone.head_id.weight.data
            delattr(self.backbone.backbone, "head_id")
            softmax_weights_norm = torch.norm(self.softmax_weights, dim=1, keepdim=True)
            self.softmax_weights = (
                self.softmax_weights / softmax_weights_norm * scf_loss.radius
            )
            self.softmax_weights = torch.nn.Parameter(
                self.softmax_weights, requires_grad=False
            )
        else:
            self.softmax_weights = softmax_weights.softmax_weights
        self.softmax_weights = softmax_weights.softmax_weights

        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.permute_batch = permute_batch
        self.validation_step_outputs = []
        self.validation_dataset = validation_dataset
        self.template_pooling_strategy = template_pooling_strategy
        self.recognition_method = recognition_method
        self.verification_metrics = verification_metrics
        self.verification_uncertainty_metrics = verification_uncertainty_metrics
        self.predict_kappa_by_input = predict_kappa_by_input
        self.backbone_input = backbone_input

    def forward(self, x):
        self.backbone.eval()
        backbone_outputs = self.backbone(x)
        if self.predict_kappa_by_input:
            # x = torch.flatten(x, 1)
            bottleneck_feature = self.backbone_input(x)["feature"]
            # bottleneck_feature = self.backbone_input(x)["bottleneck_feature"]
            log_kappa = self.head({"bottleneck_feature": bottleneck_feature})
        else:
            log_kappa = self.head(backbone_outputs)
        return backbone_outputs["feature"], log_kappa

    def training_step(self, batch):
        images, labels = batch
        # freezing bn layers
        feature, log_kappa = self(images)
        kappa = torch.exp(log_kappa)
        # print(labels)
        # print(feature[0] @ self.softmax_weights[labels[0]] / 32)
        wc = self.softmax_weights[labels, :]
        losses, l1, l2, l3, cos = self.scf_loss(feature, kappa, wc)

        kappa_mean = kappa.mean()
        total_loss = losses.mean()

        self.log("train_loss", total_loss.item(), prog_bar=True)
        self.log("kappa", kappa_mean.item())
        self.log("l1", l1.mean().item())
        self.log("l2", l2.mean().item())
        self.log("l3", l3.mean().item())
        self.log("cos", cos.mean().item())

        return total_loss

    def configure_optimizers(self):
        if self.predict_kappa_by_input:
            params = [*self.head.parameters()] + [*self.backbone_input.parameters()]
        else:
            params = [*self.head.parameters()]
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
        features, log_kappa = self(images)
        self.validation_step_outputs.append([features, log_kappa, labels])

    def on_validation_epoch_end(self):
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

        unc_indexes = np.argsort(-kappa[:, 0])
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
        # test_dataset = self.validation_dataset
        # pooled_data = self.template_pooling_strategy(
        #     image_input_feats,
        #     unc,
        #     test_dataset.templates,
        #     test_dataset.medias,
        # )
        # pooling_default_strategy = PoolingDefault()
        # pooled_default_data = pooling_default_strategy(
        #     image_input_feats,
        #     unc,
        #     test_dataset.templates,
        #     test_dataset.medias,
        # )

        # template_pooled_emb = pooled_data[0]
        # template_pooled_unc = pooled_data[1]
        # template_pooled_default_emb = pooled_default_data[0]
        # template_pooled_default_unc = pooled_default_data[1]
        # template_ids = np.unique(test_dataset.templates)
        # scores, unc = self.recognition_method(
        #     template_pooled_emb,
        #     template_pooled_unc,
        #     template_ids,
        #     test_dataset.p1,
        #     test_dataset.p2,
        # )
        # scores_default, unc_default = self.recognition_method(
        #     template_pooled_default_emb,
        #     template_pooled_default_unc,
        #     template_ids,
        #     test_dataset.p1,
        #     test_dataset.p2,
        # )
        # print(scores.shape)
        # metrics = {}
        # for metric in self.verification_metrics:
        #     print(metric)
        #     metrics.update(
        #         metric(
        #             scores=scores,
        #             labels=test_dataset.label,
        #         )
        #     )
        # print(metrics)
        # unc_metrics = {}

        # # compute uncertainty metrics
        # for unc_metric in self.verification_uncertainty_metrics:
        #     print(unc_metric)
        #     unc_metrics.update(
        #         unc_metric(
        #             scores=scores_default,
        #             labels=test_dataset.label,
        #             predicted_unc=unc_default[:, 0],
        #         )
        #     )
        # print(unc_metrics)
        # for metric_name, value in metrics.items():
        #     if "TAR" in metric_name:
        #         self.log(metric_name, value)

        # for unc_metric_name, unc_value in unc_metrics.items():
        #     if "TAR" in unc_metric_name:
        #         filter_auc = unc_metrics["fractions"][-1] * np.mean(unc_value)
        #         name = unc_metric_name.split(":")[-1]
        #         self.log(f"filter_auc_{name}", filter_auc)
