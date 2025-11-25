import torch
import torch.nn.functional as F
from training import models as mlib
from training.models.whale_arcface.src.train import SphereClassifier
from transformers import AutoModel
import torch
import torch.nn as nn


class EmbModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


class EfficientNet(torch.nn.Module):
    def __init__(self, checkpoint_path: str, learnable: bool) -> None:
        super().__init__()
        self.backbone = SphereClassifier.load_from_checkpoint(
            checkpoint_path=checkpoint_path
        )
        delattr(self.backbone, "head_species")
        if learnable is False:
            for p in self.backbone.modules():
                p.requires_grad = False

    def forward(self, x):
        bottleneck_feat = self.backbone.get_bottleneck_feature(x)
        feats = self.backbone.backbone_head_bn(
            self.backbone.backbone_head(bottleneck_feat)
        )
        feats = F.normalize(feats, p=2.0, dim=1)
        return {"bottleneck_feature": bottleneck_feat, "feature": feats}


class ResNet(torch.nn.Module):
    def __init__(self, resnet_name: str, weights, use_cpu=False, **kwargs) -> None:
        super().__init__()
        self.backbone = mlib.model_dict[resnet_name](**kwargs)

        if weights is not False:
            if use_cpu:
                backbone_dict = torch.load(weights, map_location=torch.device("cpu"))
            else:
                backbone_dict = torch.load(weights)
            self.backbone.load_state_dict(backbone_dict)

    def forward(self, x):
        return self.backbone(x)


class BERTEmbedder(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_features: int = 768,  # final embedding dim (on sphere)
        bottleneck_dim: int = 768,  # dim before final projection (often same as num_features)
        projection: bool = False,  # optional linear projection layer
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.bottleneck_dim = bottleneck_dim
        self.backbone = AutoModel.from_pretrained(model_name)

        # Optional linear head (common in metric learning)
        if projection:
            self.proj = nn.Linear(self.backbone.config.hidden_size, bottleneck_dim)
            self.final_proj = nn.Linear(bottleneck_dim, num_features)
        else:
            # Use identity (directly use [CLS])
            self.proj = nn.Identity()
            self.final_proj = nn.Identity()
            assert bottleneck_dim == num_features == self.backbone.config.hidden_size

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.proj.parameters():
                param.requires_grad = False

    def forward(self, batch: dict):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # [CLS] token representation: (B, hidden_size)
        cls_emb = outputs.last_hidden_state[:, 0]  # (B, 768)

        # Bottleneck feature (before final norm/projection)
        bottleneck_feat = self.proj(cls_emb)  # (B, bottleneck_dim)

        # Final feature (for ArcFace): L2-normalized on hypersphere
        final_feat = self.final_proj(bottleneck_feat)  # (B, num_features)
        final_feat = F.normalize(final_feat, p=2, dim=1)

        return {
            "bottleneck_feature": bottleneck_feat,  # for SCF confidence κ(x)
            "feature": final_feat,  # for ArcFace classification
        }
