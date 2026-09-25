"""PyTorch Dataset + transform builders for the LIDC-IDRI canonical manifest."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["Benign", "Malignant", "Normal"]  # alphabetical, fixed order used everywhere
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def build_transforms(image_size: int = 224, train: bool = True, augment_strength: str = "medium") -> v2.Compose:
    """Geometric + intensity augmentation: rotation, horizontal flip, zoom/scale,
    translation, brightness/contrast. Applied ONLY to the training split --
    validation/test always use the deterministic pipeline.
    """
    if not train:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((image_size, image_size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    # "ct" keeps only the variation a chest CT can actually show. Hounsfield
    # units are physically calibrated, so a grey level means a tissue density:
    # jittering brightness/contrast corrupts signal rather than nuisance. And a
    # nodule spans a few pixels of a 512px slice, so translation and rescaling
    # can resample it away. Flip and a small rotation survive both objections.
    strengths = {
        "ct": dict(rot=7, jitter=0.0, translate=0.0, scale=(1.0, 1.0)),
        "light": dict(rot=10, jitter=0.1, translate=0.05, scale=(0.95, 1.05)),
        "medium": dict(rot=15, jitter=0.2, translate=0.1, scale=(0.9, 1.1)),
        "heavy": dict(rot=20, jitter=0.3, translate=0.15, scale=(0.85, 1.15)),
    }
    p = strengths[augment_strength]
    steps = [
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
    ]
    if p["translate"] or p["scale"] != (1.0, 1.0):
        steps.append(v2.RandomAffine(degrees=p["rot"],
                                     translate=(p["translate"], p["translate"]),
                                     scale=p["scale"]))
    elif p["rot"]:
        steps.append(v2.RandomRotation(degrees=p["rot"]))
    if p["jitter"]:
        steps.append(v2.ColorJitter(brightness=p["jitter"], contrast=p["jitter"]))
    steps += [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return v2.Compose(steps)


class LungCTDataset(Dataset):
    """Reads rows from one of the manifest CSVs produced by src/audit/*.py."""

    def __init__(self, df: pd.DataFrame, transform: v2.Compose, label_col: str = "canonical_label"):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with Image.open(row["path"]) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        y = CLASS_TO_IDX[row[self.label_col]]
        return x, y
