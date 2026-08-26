from pathlib import Path

import timm
import torch.nn as nn


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    drop_rate: float = 0.3,
    drop_path_rate: float = 0.2,
    model_name: str = "efficientnet_b0",
) -> nn.Module:
    # drop_rate = dropout di classifier head, drop_path_rate = stochastic depth
    # di backbone — dua-duanya regularisasi arsitektur untuk mengurangi overfitting.
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )


class RadImageNetClassifier(nn.Module):
    """ResNet50 pretrained di RadImageNet (citra medis) alih-alih ImageNet.

    Backbone dari paket tidak resmi `radimagenet-models` (port dari bobot Keras
    resmi RadImageNet). Atribut classifier diberi nama `classifier` supaya
    freeze_backbone/unfreeze_all di bawah tetap berfungsi tanpa modifikasi.
    """

    def __init__(self, num_classes: int = 2, drop_rate: float = 0.3, weights_path: str | Path = None):
        super().__init__()
        from radimagenet_models.models.resnet import radimagenet_resnet50

        kwargs = {"model_path": str(weights_path)} if weights_path else {}
        self.backbone = radimagenet_resnet50(**kwargs)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


def build_radimagenet_model(num_classes: int = 2, drop_rate: float = 0.3, weights_path=None) -> nn.Module:
    return RadImageNetClassifier(num_classes=num_classes, drop_rate=drop_rate, weights_path=weights_path)


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("Tidak ada layer classifier yang ditemukan untuk di-unfreeze")


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
