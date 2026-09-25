"""Cache the stacking meta-features to disk, once.

Every meta-learner experiment needs the same two arrays: the out-of-fold
probabilities the base models produce on their own validation splits, and the
per-fold probabilities they produce on the test set. Recomputing those means a
full GPU pass over 3101 images at 512px, which is minutes per experiment for
data that never changes. Caching them turns meta-learner tuning into CPU work.

The out-of-fold rule is the whole point: fold k's validation rows are scored
only by fold k's own two models, which never saw them during training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import CLASS_NAMES, build_transforms
from evaluation.stacking_ensemble import VARIANTS, probs_for_model
from inference.predictor import EnsemblePredictor

REPORT_DIR = Path("D:/skripsi/Kodingan/outputs/reports")
N_FOLDS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="full")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--augment-strength", default="medium",
                         choices=["ct", "light", "medium", "heavy"],
                         help="Which augmentation preset the checkpoints were trained with.")
    ap.add_argument("--archs", nargs="+", default=["efficientnet_b0", "resnet50"],
                    help="Base architectures; the meta-feature width is 3 x len(archs).")
    ap.add_argument("--tag", default="", help="Suffix so a wider run does not overwrite a narrower one.")
    args = ap.parse_args()

    trainval_csv, test_csv, model_dir, suffix, img_size = VARIANTS[args.variant]
    if args.no_augment:
        model_dir = model_dir.with_name(model_dir.name + "_noaug")
        suffix = suffix + "_noaug"
    elif args.augment_strength != "medium":
        model_dir = model_dir.with_name(f"{model_dir.name}_aug{args.augment_strength}")
        suffix = suffix + f"_aug{args.augment_strength}"

    trainval = pd.read_csv(trainval_csv)
    test = pd.read_csv(test_csv)
    l2i = {c: i for i, c in enumerate(CLASS_NAMES)}
    transform = build_transforms(img_size, train=False)
    predictor = EnsemblePredictor(model_dir, img_size=img_size)
    print(f"Variant: {args.variant}{'  (tanpa augmentasi)' if args.no_augment else ''} | "
          f"img_size={img_size} | {len(predictor.models)} model", flush=True)

    width = 3 * len(args.archs)
    # out-of-fold features, in the manifest's own row order
    oof_X = np.zeros((len(trainval), width))
    for k in range(N_FOLDS):
        rows = np.where(trainval["fold"].values == k)[0]
        paths = trainval["path"].values[rows]
        parts = [probs_for_model(predictor, f"{a}_fold{k}", paths, transform) for a in args.archs]
        oof_X[rows] = np.concatenate(parts, axis=1)
        print(f"  out-of-fold {k}: {len(rows)} sampel", flush=True)
    oof_y = trainval["canonical_label"].map(l2i).values

    # test features, kept per fold so the meta-learner can be applied fold-wise
    test_X = np.zeros((N_FOLDS, len(test), width))
    for k in range(N_FOLDS):
        parts = [probs_for_model(predictor, f"{a}_fold{k}", test["path"].values, transform)
                 for a in args.archs]
        test_X[k] = np.concatenate(parts, axis=1)
        print(f"  uji fold {k}: selesai", flush=True)
    test_y = test["canonical_label"].map(l2i).values

    out = REPORT_DIR / f"meta_features{suffix}{args.tag}.npz"
    np.savez_compressed(out, oof_X=oof_X, oof_y=oof_y, oof_fold=trainval["fold"].values,
                        oof_group=trainval["split_group"].values.astype(str),
                        oof_origin=trainval["label_origin"].values.astype(str),
                        test_X=test_X, test_y=test_y,
                        test_source=test["source_dataset"].values.astype(str),
                        archs=np.array(args.archs))
    print(f"\nTersimpan: {out.name}  (oof {oof_X.shape}, uji {test_X.shape})")


if __name__ == "__main__":
    main()
