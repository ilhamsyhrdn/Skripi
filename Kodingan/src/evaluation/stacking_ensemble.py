"""Stacking ensemble: a meta-learner (multinomial logistic regression) learns
the best way to combine EfficientNet-B0 + ResNet50 probabilities, instead of
plain/weighted averaging. Matches the ensemble-stacking approach reported by
Noman et al. (2025), LungCT-NET.

Leakage-safe design: the meta-learner is trained ONLY on out-of-fold
predictions -- for each fold k, the val split of fold k was never seen by
that fold's own efficientnet_b0_fold{k} / resnet50_fold{k} models during
their own training, so predicting on it with exactly those two models is a
clean out-of-fold signal. At test time, the meta-learner is applied once per
fold's (effnet, resnet) model pair and the 5 outputs are averaged, mirroring
how the plain ensemble already averages across folds.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import CLASS_NAMES, build_transforms
from inference.predictor import EnsemblePredictor

MANIFESTS = Path("D:/skripsi/Kodingan/outputs/manifests")
OUTPUTS = Path("D:/skripsi/Kodingan/outputs")
REPORT_DIR = OUTPUTS / "reports"

# variant -> (trainval csv, test csv, model dir, filename suffix, img size)
VARIANTS = {
    "combined": (MANIFESTS / "trainval_folds_combined.csv",
                 MANIFESTS / "test_holdout_combined.csv",
                 OUTPUTS / "models_combined", "_combined", 224),
    "full": (MANIFESTS / "trainval_folds_full.csv",
             MANIFESTS / "test_holdout_full.csv",
             OUTPUTS / "models_full", "_full", 512),
}

MALIGNANT_IDX = CLASS_NAMES.index("Malignant")
N_FOLDS = 5
THRESHOLDS = [0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]


@torch.no_grad()
def probs_for_model(predictor, model_key, image_paths, transform):
    model = predictor.models[model_key]
    out = np.zeros((len(image_paths), 3))
    for i, p in enumerate(image_paths):
        with Image.open(p) as im:
            x = transform(im.convert("RGB")).unsqueeze(0).to(predictor.device)
        out[i] = torch.softmax(model(x).float(), dim=1)[0].cpu().numpy()
    return out


def predict_with_threshold(probs, t):
    y_pred = probs.argmax(1)
    override = probs[:, MALIGNANT_IDX] >= t
    return np.where(override, MALIGNANT_IDX, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="combined")
    parser.add_argument("--no-augment", action="store_true",
                         help="Evaluate the ablation models trained without augmentation.")
    parser.add_argument("--augment-strength", default="medium",
                         choices=["ct", "light", "medium", "heavy"],
                         help="Which augmentation preset the checkpoints were trained with.")
    args = parser.parse_args()
    trainval_csv, test_csv, model_dir, suffix, img_size = VARIANTS[args.variant]
    if args.no_augment:
        model_dir = model_dir.with_name(model_dir.name + "_noaug")
        suffix = suffix + "_noaug"
    elif args.augment_strength != "medium":
        model_dir = model_dir.with_name(f"{model_dir.name}_aug{args.augment_strength}")
        suffix = suffix + f"_aug{args.augment_strength}"

    predictor = EnsemblePredictor(model_dir, img_size=img_size)
    transform = build_transforms(img_size, train=False)
    trainval = pd.read_csv(trainval_csv)
    test_df = pd.read_csv(test_csv)
    label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

    print(f"Variant: {args.variant} | img_size={img_size} | "
          f"{len(predictor.member_keys)} model dimuat", flush=True)
    print("Membangun fitur out-of-fold untuk meta-learner...", flush=True)
    meta_X, meta_y = [], []
    for k in range(N_FOLDS):
        val_df = trainval[trainval["fold"] == k]
        eff_probs = probs_for_model(predictor, f"efficientnet_b0_fold{k}", val_df["path"].tolist(), transform)
        res_probs = probs_for_model(predictor, f"resnet50_fold{k}", val_df["path"].tolist(), transform)
        feats = np.concatenate([eff_probs, res_probs], axis=1)  # (n, 6)
        meta_X.append(feats)
        meta_y.append(val_df["canonical_label"].map(label_to_idx).values)
        print(f"  fold {k}: {len(val_df)} sampel out-of-fold", flush=True)

    meta_X = np.concatenate(meta_X, axis=0)
    meta_y = np.concatenate(meta_y, axis=0)
    print(f"Total sampel pelatihan meta-learner: {len(meta_X)}", flush=True)

    meta_learner = LogisticRegression(max_iter=2000)
    meta_learner.fit(meta_X, meta_y)
    with open(REPORT_DIR / f"stacking_meta_learner{suffix}.pkl", "wb") as f:
        pickle.dump(meta_learner, f)
    print("Meta-learner terlatih.", flush=True)

    print("\nMenerapkan stacking ke data uji...", flush=True)
    y_true = test_df["canonical_label"].map(label_to_idx).values
    fold_meta_probs = []
    for k in range(N_FOLDS):
        eff_probs = probs_for_model(predictor, f"efficientnet_b0_fold{k}", test_df["path"].tolist(), transform)
        res_probs = probs_for_model(predictor, f"resnet50_fold{k}", test_df["path"].tolist(), transform)
        feats = np.concatenate([eff_probs, res_probs], axis=1)
        fold_meta_probs.append(meta_learner.predict_proba(feats))
        print(f"  fold {k} selesai", flush=True)

    final_probs = np.mean(fold_meta_probs, axis=0)
    np.save(REPORT_DIR / f"stacking_probs{suffix}.npy", final_probs)

    rows = []
    for t in THRESHOLDS:
        y_pred = predict_with_threshold(final_probs, t)
        acc = float(np.mean(y_pred == y_true))
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        cancer_recall = recall_score(y_true, y_pred, labels=[MALIGNANT_IDX], average="macro", zero_division=0)
        cancer_precision = precision_score(y_true, y_pred, labels=[MALIGNANT_IDX], average="macro", zero_division=0)
        rows.append({"threshold": t, "accuracy": acc, "macro_f1": macro_f1,
                     "cancer_recall": cancer_recall, "cancer_precision": cancer_precision})
        print(f"[Stacking] t={t:.2f}  acc={acc:.4f}  macroF1={macro_f1:.4f}  "
              f"cancerRecall={cancer_recall:.4f}  cancerPrecision={cancer_precision:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(REPORT_DIR / f"stacking_threshold_sweep{suffix}.csv", index=False)
    print(f"\nTersimpan: stacking_threshold_sweep{suffix}.csv", flush=True)


if __name__ == "__main__":
    main()
