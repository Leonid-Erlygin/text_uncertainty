import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import BasePredictionWriter
import importlib
import pickle
from pathlib import Path
import numpy as np
import sys

from .heads import PFEHeadAdjustableLightning

sys.path.append("/app")
from training import models as mlib


class IJB_writer(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval: str, subset: str):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.subset = subset

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        embs = torch.cat([batch[0] for batch in predictions], axis=0).numpy()
        unc = torch.cat([batch[1] for batch in predictions], axis=0).numpy()
        print(embs.shape, unc.shape)
        np.savez(
            self.output_dir / f"pfe_ijb_embs_{self.subset}.npz", embs=embs, unc=unc
        )
        # [word for sentence in text for word in sentence]


class ProbabilisticFaceEmbedding(LightningModule):
    def __init__(
        self,
        weights: str,
        backbone: torch.nn.Module,
        head: torch.nn.Module,  # head_args,
        pfe_loss: torch.nn.Module,
        optimizer_params,
        scheduler_params,
    ):
        super().__init__()

        self.backbone = (
            backbone  # mlib.model_dict["iresnet50_normalized"](learnable=False)
        )
        self.head = head  # PFEHeadAdjustableLightning(**head_args)
        if weights != "None":
            checkpoint = torch.load(weights)
            self.backbone = mlib.model_dict["iresnet50_normalized"](learnable=False)
            self.backbone.load_state_dict(checkpoint["backbone"])
            self.head.load_state_dict(checkpoint["head"])
        self.pfe_loss = pfe_loss
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

    def forward(self, x):
        backbone_outputs = self.backbone(x)
        log_sigma = self.head(backbone_outputs["bottleneck_feature"])
        return backbone_outputs["feature"], log_sigma

    def training_step(self, batch):
        images, labels = batch
        feature, log_sigma_sq = self(images)
        loss = self.pfe_loss(feature, labels, log_sigma_sq)
        self.log("train_loss", loss.item(), prog_bar=True)

        self.log("log_sigma_sq", log_sigma_sq.mean().item())

        return loss

    def configure_optimizers(self):
        optimizer = getattr(
            importlib.import_module(self.optimizer_params["optimizer_path"]),
            self.optimizer_params["optimizer_name"],
        )(self.head.parameters(), **self.optimizer_params["params"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": getattr(
                    importlib.import_module("torch.optim.lr_scheduler"),
                    self.scheduler_params["scheduler"],
                )(optimizer, **self.scheduler_params["params"]),
                "interval": "step",
            },
        }

    def predict_step(self, batch, batch_idx):
        images_batch = batch
        images_batch = images_batch.permute(0, 3, 1, 2)

        return self(images_batch)
