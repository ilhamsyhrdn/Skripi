from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torchvision import datasets, transforms

IMG_SIZE = 224  # default, dipakai kalau get_transforms/get_dataloaders dipanggil tanpa img_size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool, img_size: int = IMG_SIZE) -> transforms.Compose:
    # Augmentasi dipilih sesuai anatomi CT paru: TIDAK ada vertical flip (paru
    # punya orientasi atas-bawah yang bermakna) dan TIDAK ada rotasi ekstrim atau
    # color jitter berlebihan (bisa membuat citra tidak realistis / merusak fitur).
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_dataloaders(data_root, batch_size: int = 32, num_workers: int = 0, img_size: int = IMG_SIZE):
    data_root = Path(data_root)

    train_ds = datasets.ImageFolder(data_root / "train", transform=get_transforms(True, img_size))
    val_ds = datasets.ImageFolder(data_root / "val", transform=get_transforms(False, img_size))
    test_ds = datasets.ImageFolder(data_root / "test", transform=get_transforms(False, img_size))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_ds.classes


def compute_class_weights(image_folder_dataset, device) -> torch.Tensor:
    targets = np.array(image_folder_dataset.targets)
    classes = np.unique(targets)
    weights = compute_class_weight("balanced", classes=classes, y=targets)
    return torch.tensor(weights, dtype=torch.float32, device=device)
