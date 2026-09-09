"""PyTorch Dataset + transform builders for the deduplicated lung-CT manifest."""
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
    """Basic (geometric + intensity) augmentation per Chlap et al. (2021) and the proposal's
    section 2.5 / 3.2: rotation, horizontal flip, zoom/scale, translation, brightness/contrast.
    Applied ONLY to the training split -- validation/test always use the deterministic pipeline.
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

    strengths = {
        "light": dict(rot=10, jitter=0.1, translate=0.05, scale=(0.95, 1.05)),
        "medium": dict(rot=15, jitter=0.2, translate=0.1, scale=(0.9, 1.1)),
        "heavy": dict(rot=20, jitter=0.3, translate=0.15, scale=(0.85, 1.15)),
    }
    p = strengths[augment_strength]
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=p["rot"], translate=(p["translate"], p["translate"]), scale=p["scale"]),
            v2.ColorJitter(brightness=p["jitter"], contrast=p["jitter"]),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


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

    @property
    def targets(self):
        return [CLASS_TO_IDX[v] for v in self.df[self.label_col]]


def load_manifest(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)
