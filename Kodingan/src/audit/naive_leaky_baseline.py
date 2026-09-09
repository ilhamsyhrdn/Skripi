"""
Deliberately reproduces the "naive" mistake so it can be quantified: take every labeled,
non-synthetic image AS-IS (including all cross-dataset duplicates and near-duplicate
crops/slices from the same patient), split it 70/15/15 **per image, purely at random**
(the way a first pass at this problem usually gets done), and train one single
EfficientNet-B0 with the exact same schedule as the real pipeline.

This is NOT the model used anywhere in the thesis results -- it exists purely to produce
an honest, reproducible before/after comparison for `Kejanggalan Dataset/08_bukti_dampak_leakage.md`.

Run:  Kodingan/.venv/Scripts/python.exe -m src.audit.naive_leaky_baseline
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

KODINGAN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KODINGAN_DIR))

from src.data.dataset import CLASS_NAMES, LungCTDataset, build_transforms  # noqa: E402
from src.models.factory import build_model, freeze_backbone, unfreeze_for_finetune  # noqa: E402
from src.training.engine import fit  # noqa: E402
from src.training.train_cv import class_weights_from_df  # noqa: E402

MANIFEST_DIR = KODINGAN_DIR / "outputs" / "manifests"
REPORT_DIR = KODINGAN_DIR / "outputs" / "reports"
SEED = 42


def main():
    full = pd.read_csv(MANIFEST_DIR / "full_manifest.csv")
    naive_pool = full[full["is_labeled"] & (~full["is_synthetic_source"]) & (~full["corrupt"])].copy()
    print(f"Naive pool (every labeled non-synthetic image, duplicates included): {len(naive_pool)}")
    print(naive_pool["canonical_label"].value_counts())

    train_df, temp_df = train_test_split(
        naive_pool, test_size=0.30, stratify=naive_pool["canonical_label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["canonical_label"], random_state=SEED
    )
    print(f"Naive random split (per-image, no grouping): train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # How much of the "held-out" naive test set is actually a duplicate of a naive-train image?
    train_groups = set(train_df["group_id"])
    test_leaked = test_df["group_id"].isin(train_groups).mean()
    val_leaked = val_df["group_id"].isin(train_groups).mean()
    print(f"Fraction of naive TEST images whose duplicate-group also appears in naive TRAIN: {test_leaked:.3f}")
    print(f"Fraction of naive VAL images whose duplicate-group also appears in naive TRAIN:  {val_leaked:.3f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = build_transforms(224, train=True, augment_strength="medium")
    eval_tf = build_transforms(224, train=False)
    train_loader = DataLoader(LungCTDataset(train_df, train_tf), batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(LungCTDataset(val_df, eval_tf), batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(LungCTDataset(test_df, eval_tf), batch_size=32, shuffle=False, num_workers=0)

    torch.manual_seed(SEED)
    model = build_model("efficientnet_b0").to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_df(train_df, device))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_holder = {"best_f1": -1.0}

    freeze_backbone(model, "efficientnet_b0")
    opt_a = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    fit(model, train_loader, val_loader, criterion, opt_a, device, epochs=12, phase_name="A-head",
        scaler=scaler, patience=6, best_state_holder=best_holder)

    model.load_state_dict(best_holder["state_dict"])
    unfreeze_for_finetune(model, "efficientnet_b0", n_blocks=3)
    opt_b = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_b, mode="max", factor=0.5, patience=3)
    fit(model, train_loader, val_loader, criterion, opt_b, device, epochs=25, phase_name="B-finetune",
        scaler=scaler, scheduler=sched, patience=6, best_state_holder=best_holder)

    model.load_state_dict(best_holder["state_dict"])

    from src.training.engine import run_epoch
    test_res = run_epoch(model, test_loader, criterion, device, None, scaler)
    print(f"\nNAIVE (leaky) single-model TEST result: acc={test_res.accuracy:.3f} "
          f"macroF1={test_res.macro_f1:.3f} malignantRecall={test_res.malignant_recall:.3f}")

    with open(REPORT_DIR / "naive_leaky_baseline.json", "w", encoding="utf-8") as f:
        json.dump({
            "naive_pool_size": len(naive_pool),
            "train_size": len(train_df), "val_size": len(val_df), "test_size": len(test_df),
            "fraction_test_images_duplicate_group_also_in_train": float(test_leaked),
            "fraction_val_images_duplicate_group_also_in_train": float(val_leaked),
            "test_accuracy": test_res.accuracy,
            "test_macro_f1": test_res.macro_f1,
            "test_malignant_recall": test_res.malignant_recall,
            "test_per_class_recall": test_res.per_class_recall,
        }, f, indent=2)


if __name__ == "__main__":
    main()
