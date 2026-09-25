"""Build the OOD (out-of-distribution) manifest: LIDC-IDRI canonical images
as the positive "Lung CT" class, and an equal-sized random sample of COCO
val2017 as the negative "Bukan Lung CT" class, using the SAME train/val/test
split proportions as the main classification pool (bab 3.5.3 design).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

POOL_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/lidc_canonical_pool.csv")
TEST_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/test_holdout.csv")
TRAINVAL_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/trainval_folds.csv")
COCO_DIR = Path("D:/skripsi/Dataset/Dataset COCO/val2017")
OUT_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/ood_manifest.csv")
SEED = 42


def main():
    test_df = pd.read_csv(TEST_CSV)
    trainval_df = pd.read_csv(TRAINVAL_CSV)
    n_val = max(1, round(len(trainval_df) * 0.10))

    pos_rows = []
    for _, r in test_df.iterrows():
        pos_rows.append({"path": r["path"], "label": 1, "class_name": "Lung_CT", "split": "test"})
    # simple 90/10 train/val split of trainval for the OOD positive class
    shuffled = trainval_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_part, train_part = shuffled.iloc[:n_val], shuffled.iloc[n_val:]
    for _, r in train_part.iterrows():
        pos_rows.append({"path": r["path"], "label": 1, "class_name": "Lung_CT", "split": "train"})
    for _, r in val_part.iterrows():
        pos_rows.append({"path": r["path"], "label": 1, "class_name": "Lung_CT", "split": "val"})

    n_total_pos = len(pos_rows)
    coco_files = sorted(COCO_DIR.glob("*.jpg"))
    if len(coco_files) < n_total_pos:
        raise RuntimeError(f"Not enough COCO images: found {len(coco_files)}, need {n_total_pos}")

    import random
    rng = random.Random(SEED)
    sampled = coco_files.copy()
    rng.shuffle(sampled)
    sampled = sampled[:n_total_pos]

    n_test_neg = len(test_df)
    n_val_neg = n_val
    neg_rows = []
    idx = 0
    for _ in range(n_test_neg):
        neg_rows.append({"path": str(sampled[idx]), "label": 0, "class_name": "Bukan_Lung_CT", "split": "test"}); idx += 1
    for _ in range(n_val_neg):
        neg_rows.append({"path": str(sampled[idx]), "label": 0, "class_name": "Bukan_Lung_CT", "split": "val"}); idx += 1
    while idx < len(sampled):
        neg_rows.append({"path": str(sampled[idx]), "label": 0, "class_name": "Bukan_Lung_CT", "split": "train"}); idx += 1

    df = pd.concat([pd.DataFrame(pos_rows), pd.DataFrame(neg_rows)], ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(df.groupby(["split", "class_name"]).size(), flush=True)
    print(f"Saved: {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
