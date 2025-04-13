import torch
import math
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import BasePredictionWriter, Callback
import importlib
from pathlib import Path
import numpy as np
from evaluation.evaluate import instantiate_list
from evaluation.template_pooling_strategies import PoolingDefault
import torch.nn.functional as F


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
        self.softmax_weights = torch.load(softmax_weights_path)
        softmax_weights_norm = torch.norm(
            self.softmax_weights, dim=1, keepdim=True
        )  # [N, 512]
        self.softmax_weights = (
            self.softmax_weights / softmax_weights_norm * radius
        )  # $ w_c \in rS^{d-1} $

        self.softmax_weights = torch.nn.Parameter(
            self.softmax_weights, requires_grad=requires_grad
        )


class ScaleFace(LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        scaleface_loss: torch.nn.Module,
        optimizer_params,
        scheduler_params,
        softmax_weights: torch.nn.Module,
        permute_batch: bool,
        validation_dataset=None,
        template_pooling_strategy=None,
        recognition_method=None,
        verification_metrics=None,
        verification_uncertainty_metrics=None,
        predict_scale_by_input=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        self.head = head
        self.scaleface_loss = scaleface_loss
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
        self.predict_scale_by_input = predict_scale_by_input

    def forward(self, x):
        self.backbone.eval()
        backbone_outputs = self.backbone(x)
        if self.predict_scale_by_input:
            x = torch.flatten(x, 1)
            scale = self.head({"bottleneck_feature": x})
        else:
            scale = self.head(backbone_outputs)
        scale = torch.exp(scale)
        return backbone_outputs["feature"], scale

    def training_step(self, batch):
        images, labels = batch
        # freezing bn layers
        feature, scale = self(images)
        logits = F.linear(feature, self.softmax_weights)
        loss = self.scaleface_loss(
            logits, labels, scale
        )  # losses, l1, l2, l3, cos = self.scaleface_loss(logits, labels, scale)

        scale_mean = scale.mean()
        total_loss = loss.mean()

        self.log("train_loss", total_loss.item(), prog_bar=True)
        self.log("scale", scale_mean.item())
        return total_loss

    def configure_optimizers(self):
        optimizer = getattr(
            importlib.import_module(self.optimizer_params["optimizer_path"]),
            self.optimizer_params["optimizer_name"],
        )(
            [*self.head.parameters()],
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
