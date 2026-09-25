"""Same as make_splits_crop.py but for the pathology-corrected final pool
(combined_canonical_pool.csv)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

IN_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/combined_canonical_pool.csv")
TEST_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/test_holdout_combined.csv")
TRAINVAL_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/trainval_folds_combined.csv")
N_FOLDS = 5
TEST_FRAC = 0.18
SEED = 42


def main():
    df = pd.read_csv(IN_CSV)
    print(f"Canonical pool (final): {len(df)} images, {df['split_group'].nunique()} patients", flush=True)
    print(df["canonical_label"].value_counts(), flush=True)

    rng = np.random.RandomState(SEED)
    patient_label = df.groupby("split_group")["canonical_label"].agg(lambda s: s.value_counts().idxmax())

    test_groups = set()
    for label, group_ids in patient_label.groupby(patient_label).groups.items():
        group_ids = list(group_ids)
        rng.shuffle(group_ids)
        n_test = max(1, round(len(group_ids) * TEST_FRAC))
        test_groups.update(group_ids[:n_test])

    test_df = df[df["split_group"].isin(test_groups)].reset_index(drop=True)
    trainval_df = df[~df["split_group"].isin(test_groups)].reset_index(drop=True)

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    trainval_df["fold"] = -1
    X = trainval_df.index.values
    y = trainval_df["canonical_label"].values
    g = trainval_df["split_group"].values
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(X, y, g)):
        trainval_df.loc[val_idx, "fold"] = fold_idx

    fold_of_group = trainval_df.groupby("split_group")["fold"].nunique()
    assert (fold_of_group == 1).all(), "A split_group leaked across folds!"
    assert not (set(trainval_df["split_group"]) & test_groups), "A split_group leaked into test!"

    TEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(TEST_CSV, index=False)
    trainval_df.to_csv(TRAINVAL_CSV, index=False)

    print(f"\nHeld-out test: {len(test_df)} images", flush=True)
    print(test_df["canonical_label"].value_counts(), flush=True)
    print(f"\nTrainval (5-fold): {len(trainval_df)} images", flush=True)
    for k in range(N_FOLDS):
        fold_df = trainval_df[trainval_df["fold"] == k]
        print(f"  fold {k}: {len(fold_df)} images -- {dict(fold_df['canonical_label'].value_counts())}", flush=True)

    print(f"\nSaved: {TEST_CSV}\nSaved: {TRAINVAL_CSV}", flush=True)


if __name__ == "__main__":
    main()
