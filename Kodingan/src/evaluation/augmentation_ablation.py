"""Compare the models trained with and without data augmentation.

Both groups share everything except the augmentation transforms, so the gap
between them is what augmentation actually bought. Two things are measured:
what it does on the held-out test set (the headline claim in Bab IV) and how
far training accuracy runs ahead of validation accuracy, which is the
overfitting that augmentation is supposed to hold back.

Run after train_cv.py --no-augment finishes, and after evaluate_models.py and
stacking_ensemble.py have been run for both groups.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

REPORTS = Path("D:/skripsi/Kodingan/outputs/reports")
MANIFESTS = Path("D:/skripsi/Kodingan/outputs/manifests")
CLASS_NAMES = ["Benign", "Malignant", "Normal"]
MAL = CLASS_NAMES.index("Malignant")
VARIANT = "full"


def stacking_metrics(suffix: str) -> dict:
    probs = np.load(REPORTS / f"stacking_probs_{suffix}.npy")
    test = pd.read_csv(MANIFESTS / f"test_holdout_{VARIANT}.csv")
    y = test["canonical_label"].map({c: i for i, c in enumerate(CLASS_NAMES)}).values
    pred = probs.argmax(1)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "cancer_recall": recall_score(y, pred, labels=[MAL], average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, probs, multi_class="ovr", average="macro"),
    }


def overfitting_gap(prefix: str) -> dict:
    """Mean train-minus-validation accuracy at each model's best epoch."""
    gaps, train_accs, val_accs = [], [], []
    for f in sorted(REPORTS.glob(f"history_{prefix}*.json")):
        h = pd.DataFrame(json.load(open(f)))
        best = h.loc[h["macro_f1"].idxmax()]
        train_accs.append(best["train_acc"])
        val_accs.append(best["acc"])
        gaps.append(best["train_acc"] - best["acc"])
    return {"n_model": len(gaps), "train_acc": float(np.mean(train_accs)),
            "val_acc": float(np.mean(val_accs)), "gap": float(np.mean(gaps))}


def main():
    rows = []
    for label, suffix, prefix in [("tanpa", f"{VARIANT}_noaug", f"{VARIANT}_noaug_"),
                                   ("dengan", VARIANT, f"{VARIANT}_")]:
        m = stacking_metrics(suffix)
        rows.append({"konfigurasi": label, **m})
        g = overfitting_gap(prefix)
        print(f"[augmentasi {label:6s}] akurasi={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"cancerRecall={m['cancer_recall']:.4f} ROC-AUC={m['roc_auc']:.4f}", flush=True)
        print(f"{'':19s}rata-rata {g['n_model']} model pada epoch terbaik: "
              f"akurasi latih={g['train_acc']:.4f} validasi={g['val_acc']:.4f} "
              f"selisih={g['gap']:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    out = REPORTS / f"ablasi_augmentasi_{VARIANT}.csv"
    df.to_csv(out, index=False)

    d = df.set_index("konfigurasi")
    print("\nselisih (dengan - tanpa augmentasi):")
    for col in ["accuracy", "macro_f1", "cancer_recall", "roc_auc"]:
        delta = d.loc["dengan", col] - d.loc["tanpa", col]
        print(f"  {col:14s}: {delta:+.4f}")
    print(f"\nTersimpan: {out.name}")


if __name__ == "__main__":
    main()
