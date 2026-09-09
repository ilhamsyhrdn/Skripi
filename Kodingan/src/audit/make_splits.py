"""
Build a leakage-safe train / 5-fold-CV / held-out-test split from the canonical,
deduplicated image pool produced by build_manifest.py.

Two extra safety margins beyond the (already deduplicated) canonical pool:

1. `split_group`: images are additionally bucketed by
   (source_dataset, raw_folder, case_number // 5) so that nearby case indices from the
   same source folder -- which the audit showed are sometimes adjacent slices of the same
   CT study even when they didn't hash-match closely enough to be auto-merged -- always
   land in the same split. This never merges *labels*, it only constrains *where a split
   boundary can fall*, so it costs a little split randomness in exchange for a materially
   lower residual leakage risk.
2. The held-out TEST set is carved out first and never touched again; 5-fold
   StratifiedGroupKFold cross-validation runs only on the remaining train+val pool, so the
   final ensemble evaluation number is a genuine one-shot generalization estimate.

Run:  Kodingan/.venv/Scripts/python.exe Kodingan/src/audit/make_splits.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "Kodingan" / "outputs" / "manifests"
TEST_FRACTION = 0.18
N_FOLDS = 5
SEED = 42

_NUM_RE = re.compile(r"::(-?\d+(?:\.\d+)?|.+)$")


def case_bucket(case_key: str) -> str:
    m = _NUM_RE.search(case_key)
    if not m:
        return case_key
    tail = m.group(1)
    prefix = case_key[: m.start()]
    try:
        num = int(float(tail))
        return f"{prefix}::{num // 5}"
    except ValueError:
        return case_key


def main() -> None:
    pool = pd.read_csv(OUT_DIR / "canonical_pool.csv")
    pool["split_group"] = (
        pool["source_dataset"] + "|" + pool["raw_folder"] + "|" + pool["case_key"].map(case_bucket)
    )

    rng = np.random.default_rng(SEED)

    # ---- Step 1: carve out the held-out TEST set (group + label aware, greedy) ----
    group_label = pool.groupby("split_group")["canonical_label"].agg(lambda s: s.value_counts().idxmax())
    group_size = pool.groupby("split_group").size()
    groups = pd.DataFrame({"label": group_label, "size": group_size}).reset_index()

    test_groups: set[str] = set()
    for label, sub in groups.groupby("label"):
        target = int(round(sub["size"].sum() * TEST_FRACTION))
        sub = sub.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        acc = 0
        for _, row in sub.iterrows():
            if acc >= target:
                break
            test_groups.add(row["split_group"])
            acc += row["size"]

    pool["split"] = np.where(pool["split_group"].isin(test_groups), "test", "trainval")

    test_df = pool[pool["split"] == "test"].copy()
    trainval_df = pool[pool["split"] == "trainval"].copy()

    print("Held-out TEST set:")
    print(test_df["canonical_label"].value_counts())
    print("\nTrain+val pool (goes into 5-fold CV):")
    print(trainval_df["canonical_label"].value_counts())

    # ---- Step 2: StratifiedGroupKFold on trainval, respecting split_group ----
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    trainval_df = trainval_df.reset_index(drop=True)
    trainval_df["fold"] = -1
    X = trainval_df.index.values
    y = trainval_df["canonical_label"].values
    g = trainval_df["split_group"].values
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(X, y, g)):
        trainval_df.loc[val_idx, "fold"] = fold_idx

    assert (trainval_df["fold"] >= 0).all()

    print("\nPer-fold validation-label counts:")
    print(trainval_df.groupby(["fold", "canonical_label"]).size().unstack(fill_value=0))

    # Sanity check: no split_group appears in more than one fold, and none appear in test.
    fold_of_group = trainval_df.groupby("split_group")["fold"].nunique()
    assert (fold_of_group == 1).all(), "A split_group leaked across folds!"
    assert not (set(trainval_df["split_group"]) & test_groups), "A split_group leaked into test!"

    trainval_df.to_csv(OUT_DIR / "trainval_folds.csv", index=False)
    test_df.to_csv(OUT_DIR / "test_holdout.csv", index=False)

    summary = {
        "test_fraction_target": TEST_FRACTION,
        "n_folds": N_FOLDS,
        "test_counts": test_df["canonical_label"].value_counts().to_dict(),
        "trainval_counts": trainval_df["canonical_label"].value_counts().to_dict(),
        "fold_val_counts": {
            str(f): sub["canonical_label"].value_counts().to_dict()
            for f, sub in trainval_df.groupby("fold")
        },
        "n_split_groups_test": len(test_groups),
        "n_split_groups_trainval": trainval_df["split_group"].nunique(),
        "leakage_checks_passed": True,
    }
    with open(OUT_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nWrote trainval_folds.csv, test_holdout.csv, split_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
