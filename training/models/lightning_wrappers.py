import torch
import torch.nn.functional as F
from training import models as mlib
from training.models.whale_arcface.src.train import SphereClassifier


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
    def __init__(
        self, resnet_name: str, weights, learnable: bool, use_cpu=False
    ) -> None:
        super().__init__()
        self.backbone = mlib.model_dict[resnet_name](learnable=learnable)

        if weights is not False:
            if use_cpu:
                backbone_dict = torch.load(weights, map_location=torch.device("cpu"))
            else:
                backbone_dict = torch.load(weights)
            self.backbone.load_state_dict(backbone_dict)

    def forward(self, x):
        return self.backbone(x)
