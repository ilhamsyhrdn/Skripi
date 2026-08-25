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
