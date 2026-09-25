"""Model factory: EfficientNet-B0 (main) and ResNet50 (comparison), both
ImageNet-pretrained, with a Dropout(0.3) + Linear(3) classification head, and
the two-phase (feature extraction / fine-tuning) freeze helpers.

n_blocks in unfreeze_for_finetune is NOT comparable across the two: at
n_blocks=3 it reopens 99.0% of ResNet50's parameters (23.3M) but 78.8% of
EfficientNet-B0's (3.2M). The proportions are closer than the block count
suggests; what differs by an order of magnitude is the absolute capacity
being refitted, which is why the two respond so differently to fine-tuning.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)


def build_model(arch: str = "efficientnet_b0", n_classes: int = 3, dropout: float = 0.3):
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
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, n_classes))
        return model
    raise ValueError(f"unknown arch: {arch}")


def freeze_backbone(model, arch):
    """Phase A: freeze everything except the new classification head."""
    for p in model.parameters():
        p.requires_grad = False
    head = model.classifier if arch == "efficientnet_b0" else model.fc
    for p in head.parameters():
        p.requires_grad = True


def unfreeze_for_finetune(model, arch, n_blocks=3):
    """Phase B: unfreeze the last n_blocks feature blocks + head."""
    for p in model.parameters():
        p.requires_grad = False
    if arch == "efficientnet_b0":
        for block in list(model.features.children())[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif arch == "resnet50":
        for stage in [model.layer2, model.layer3, model.layer4][-n_blocks:]:
            for p in stage.parameters():
                p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True
