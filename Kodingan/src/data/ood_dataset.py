"""Dataset + transforms for the OOD (Lung CT vs Bukan Lung CT) binary module."""
from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from data.dataset import IMAGENET_MEAN, IMAGENET_STD

OOD_CLASS_NAMES = ["Bukan_Lung_CT", "Lung_CT"]


def build_ood_transforms(image_size: int = 224, train: bool = True) -> v2.Compose:
    if not train:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        v2.ColorJitter(brightness=0.15, contrast=0.15),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class OODDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(row["path"]) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, int(row["label"])
