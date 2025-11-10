import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import importlib
import pickle
import wandb
from decimal import Decimal
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader, Dataset
import math
from typing import List

from typing import Tuple, Dict


class HAAML(nn.Module):
    def __init__(
        self, in_feature=128, out_feature=10575, s=64.0, m_0=0.50, t=1.2, loss_weight=10
    ):
        super(HAAML, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m_0 = m_0
        self.t = t
        self.loss_weight = loss_weight
        self.weight = nn.Parameter(
            torch.Tensor(out_feature, in_feature)
        )  # num_class*feat_dim
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m_0)
        self.sin_m = math.sin(m_0)

    def forward(self, x, label):
        # cos(theta)
        x_norm = x.renorm(2, 0, 1e-5).mul(1e5)
        n_weight = self.weight.renorm(2, 0, 1e-5).mul(1e5)
        cos_theta = F.linear(x_norm, n_weight)
        cos_theta = cos_theta.clamp(-1, 1)

        # ground true
        batch_size = label.size(0)
        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)

        # take cos(theta + m) into cos_theta
        sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
        cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m
        # cos_theta_m = torch.cos(torch.acos(gt) + self.m_0)

        # new_m
        hard_mask = (cos_theta > cos_theta_m).type(torch.FloatTensor).cuda()
        hard_mask.scatter_(1, label.view(-1, 1), 0)
        hard_cos = torch.where(
            hard_mask > 0, cos_theta - cos_theta_m, torch.zeros_like(cos_theta)
        )
        # hard_cos_one = torch.where(hard_mask > 0, torch.ones_like(cos_theta), torch.zeros_like(cos_theta))
        hard_cos_num = torch.sum(hard_mask, dim=1).view(-1, 1)
        hard_level = torch.sum(hard_cos, dim=1).view(-1, 1)
        hard_cos_num = hard_cos_num.clamp(1, self.out_feature)  # avoid /0
        H = hard_level / hard_cos_num
        with torch.no_grad():
            new_m = self.m_0 + self.t * torch.log(H + 1)
            new_m = torch.where(new_m > 0.75, torch.zeros_like(new_m), new_m)
            cos_new_m = torch.cos(new_m)
            sin_new_m = torch.sin(new_m)

        # cos(theta + new_m)
        cos_theta_newm = gt * cos_new_m - sin_theta * sin_new_m
        # new_gt = torch.where(gt > 0, cos_theta_newm , gt) # easy_margin=true 2024.6.5

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        threshold = torch.cos(math.pi - new_m)
        mm = torch.sin(math.pi - new_m) * new_m
        new_gt = torch.where(
            gt > threshold, cos_theta_newm, gt - mm
        )  # easy_margin=false

        cos_theta.scatter_(1, label.view(-1, 1), new_gt)
        output = cos_theta * self.s

        # regularizer
        hard_regularizer = self.loss_weight * torch.mean(H)

        return output, hard_regularizer, gt, new_m.view(1, -1), H.view(1, -1)


class HAAMLModel(LightningModule):
    """Lightning wrapper for a Metric Learning model."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        num_labels: int,
        num_features: int,
        haaml_params: dict,
        scheduler_params,
        optimizer_params,
    ) -> None:
        """Initialize MetricLearningModel.

        :param backbone: core deef model to be trained
        :param loss: loss function to be used
        :param num_labels: number of target classes (people)
        :param train_set - dataset with training data
        :param val_set - dataset with test data
        :param num_features - dimensionality of the feature space
        :param batch_size, learning_rate, weight_decay - model training parameters
        :param num_workers - number of CPUs to be used (for dataloaders)
        """
        super().__init__()
        self.backbone = backbone
        self.haaml_params = haaml_params
        self.loss_function = HAAML(
            num_features,
            num_labels,
            s=haaml_params["scale_size"],
            m_0=haaml_params["m_0"],
            t=haaml_params["t"],
            loss_weight=haaml_params["loss_weight"],
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.scheduler_params = scheduler_params
        self.optimizer_params = optimizer_params

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        :param x: batch of images
        :return a tuple of:
            - features: outputs of the backbone model a.k.a. embeddings
            - logits: result of the last linear transformations
        """
        backbone_outputs = self.backbone(x)
        features = backbone_outputs["feature"]

        norm_weights = F.normalize(self.softmax_weights, dim=1)
        logits = F.linear(features, norm_weights)

        return features, logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], idx: int
    ) -> Dict[str, torch.Tensor]:
        """Do a training step of the model.

        :param batch: batch of input images and labels
        :param idx: batch number
        :return: value of the loss function
        """
        epoch = self.current_epoch
        if epoch <= 8:  # warm
            self.loss_function.t = 0.0
            self.loss_function.loss_weight = 0.0
        else:
            self.loss_function.t = self.haaml_params.t
            self.loss_function.loss_weight = self.haaml_params.loss_weight
        images, labels = batch
        raw_logits = self.backbone(images)["feature"]
        output, hard_regular, cos_theta, new_m, hardness = self.loss_function(
            raw_logits, labels
        )
        loss = self.criterion(output, labels) + hard_regular.mean()

        # log loss value
        self.log("train_loss", loss.item(), prog_bar=True)

        return {"loss": loss, "out": raw_logits, "label": labels}

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], idx: int
    ) -> Dict[str, torch.Tensor]:
        """Do a validation step of the model.

        :param batch: batch of input images and labels
        :param idx: batch number
        :return: value of the loss function
        """
        images, labels = batch
        features, logits = self(images)

        loss = self.loss(logits, labels)
        # log loss value
        self.log("val_loss", loss.item(), prog_bar=True)
        self.validation_step_outputs.append(
            {"loss": loss, "out": features, "label": labels}
        )

    def configure_optimizers(self):
        params = list(self.parameters())
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
