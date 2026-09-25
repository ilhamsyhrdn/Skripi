"""Test-Time Augmentation (TTA): average ensemble predictions over several
augmented views of each test image (original, horizontal flip, +/-10
degree rotation) instead of a single deterministic view. No retraining, no
leakage risk -- it only changes how the already-trained ensemble is queried
at inference time. Combined with the threshold sweep to find the best
achievable operating point.

--variant selects which trained ensemble to query; the TTA views are built
at that variant's training resolution, because feeding a model images at a
resolution it was not fine-tuned on degrades accuracy silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from inference.predictor import EnsemblePredictor

MANIFESTS = Path("D:/skripsi/Kodingan/outputs/manifests")
OUTPUTS = Path("D:/skripsi/Kodingan/outputs")
REPORT_DIR = OUTPUTS / "reports"

# variant -> (test csv, model dir, filename suffix, training resolution)
VARIANTS = {
    "combined": (MANIFESTS / "test_holdout_combined.csv", OUTPUTS / "models_combined", "_combined", 224),
    "full": (MANIFESTS / "test_holdout_full.csv", OUTPUTS / "models_full", "_full", 512),
}

MALIGNANT_IDX = CLASS_NAMES.index("Malignant")
THRESHOLDS = [0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]


def build_tta_views(image_size=224):
    base = [
        v2.ToImage(), v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    views = []
    views.append(v2.Compose(base))  # original
    views.append(v2.Compose([v2.ToImage(), v2.Resize((image_size, image_size), antialias=True),
                              v2.RandomHorizontalFlip(p=1.0), v2.ToDtype(torch.float32, scale=True),
                              v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]))
    for deg in (-10, 10):
        views.append(v2.Compose([v2.ToImage(), v2.Resize((image_size, image_size), antialias=True),
                                  v2.RandomRotation(degrees=(deg, deg)), v2.ToDtype(torch.float32, scale=True),
                                  v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]))
    return views


@torch.no_grad()
def predict_tta(predictor, image, views, members, weighted):
    all_probs = []
    for view_tf in views:
        x = view_tf(image.convert("RGB")).unsqueeze(0).to(predictor.device)
        probs_list = []
        for key in members:
            logits = predictor.models[key](x)
            probs_list.append(torch.softmax(logits.float(), dim=1)[0].cpu().numpy())
        if weighted:
            w = np.array([predictor.val_f1[k] for k in members]); w = w / w.sum()
        else:
            w = np.ones(len(members)) / len(members)
        ens = np.tensordot(w, np.stack(probs_list, axis=0), axes=([0], [0]))
        all_probs.append(ens)
    return np.mean(all_probs, axis=0)


def predict_with_threshold(probs, t):
    y_pred = probs.argmax(1)
    override = probs[:, MALIGNANT_IDX] >= t
    return np.where(override, MALIGNANT_IDX, y_pred)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="combined")
    args = parser.parse_args()
    test_csv, model_dir, suffix, img_size = VARIANTS[args.variant]

    test_df = pd.read_csv(test_csv)
    y_true = test_df["canonical_label"].map({c: i for i, c in enumerate(CLASS_NAMES)}).values
    predictor = EnsemblePredictor(model_dir, img_size=img_size)
    views = build_tta_views(img_size)

    print(f"Running TTA ({len(views)} views) x {len(predictor.member_keys)} models on {len(test_df)} images...", flush=True)
    probs = np.zeros((len(test_df), 3))
    for i, p in enumerate(test_df["path"]):
        with Image.open(p) as im:
            probs[i] = predict_tta(predictor, im, views, predictor.member_keys, weighted=True)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(test_df)}", flush=True)

    np.save(REPORT_DIR / f"tta_probs{suffix}.npy", probs)

    rows = []
    for t in THRESHOLDS:
        y_pred = predict_with_threshold(probs, t)
        acc = float(np.mean(y_pred == y_true))
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        cancer_recall = recall_score(y_true, y_pred, labels=[MALIGNANT_IDX], average="macro", zero_division=0)
        cancer_precision = precision_score(y_true, y_pred, labels=[MALIGNANT_IDX], average="macro", zero_division=0)
        rows.append({"threshold": t, "accuracy": acc, "macro_f1": macro_f1,
                     "cancer_recall": cancer_recall, "cancer_precision": cancer_precision})
        print(f"[TTA] t={t:.2f}  acc={acc:.3f}  macroF1={macro_f1:.3f}  "
              f"cancerRecall={cancer_recall:.3f}  cancerPrecision={cancer_precision:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / f"tta_threshold_sweep{suffix}.csv", index=False)
    print(f"\nTersimpan: tta_threshold_sweep{suffix}.csv", flush=True)


if __name__ == "__main__":
    main()
