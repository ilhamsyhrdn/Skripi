"""Model factory: EfficientNet-B0 (primary, per the proposal) + ResNet50 (second
architecture used only to add diversity to the ensemble, mirroring Wang et al. 2022 /
Saha et al. 2024 / Sandag & Kabo 2024's EfficientNet-vs-ResNet comparisons in the
attached literature).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)

N_CLASSES = 3


def build_model(arch: str = "efficientnet_b0", n_classes: int = N_CLASSES, dropout: float = 0.3) -> nn.Module:
    if arch == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, n_classes),
        )
        return model
    if arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, n_classes),
        )
        return model
    raise ValueError(f"Unknown arch: {arch}")


def freeze_backbone(model: nn.Module, arch: str) -> None:
    """Phase A: freeze everything except the new classification head (transfer learning
    with the backbone as a fixed feature extractor -- proposal section 3.3.2 step 1)."""
    for p in model.parameters():
        p.requires_grad = False
    head = model.classifier if arch == "efficientnet_b0" else model.fc
    for p in head.parameters():
        p.requires_grad = True


def unfreeze_for_finetune(model: nn.Module, arch: str, n_blocks: int = 3) -> None:
    """Phase B: unfreeze the last `n_blocks` feature blocks + the head for fine-tuning
    (proposal section 2.4 / 3.4.1 -- unfreeze deeper layers, train with a small LR)."""
    for p in model.parameters():
        p.requires_grad = False
    if arch == "efficientnet_b0":
        blocks = list(model.features.children())
        for block in blocks[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif arch == "resnet50":
        stages = [model.layer2, model.layer3, model.layer4]
        for stage in stages[-n_blocks:]:
            for p in stage.parameters():
                p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown arch: {arch}")


def trainable_param_count(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def get_last_conv_layer(model: nn.Module, arch: str) -> nn.Module:
    """Returns the last convolutional layer, used as the Grad-CAM target layer."""
    if arch == "efficientnet_b0":
        return model.features[-1]
    if arch == "resnet50":
        return model.layer4[-1]
    raise ValueError(f"Unknown arch: {arch}")
