"""Evaluate the 10 trained fold models + 4 ensemble configurations for one
experiment variant (Bab IV) on its held-out test set. Produces the CSV/JSON
tables and PNG figures: per-fold results, ensemble comparison, confusion
matrix, ROC curves, per-class metrics, and misclassification samples.

--variant selects the experiment stage: whole_slice (LIDC-IDRI, no crop) ->
crop (LIDC-IDRI, nodule crop) -> pathology (LIDC-IDRI, pathology-corrected
labels) -> combined (Kaggle + LIDC-IDRI, the adopted final configuration).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import CLASS_NAMES
from inference.predictor import EnsemblePredictor

MANIFESTS = Path("D:/skripsi/Kodingan/outputs/manifests")
OUTPUTS = Path("D:/skripsi/Kodingan/outputs")
REPORT_DIR = OUTPUTS / "reports"
FIG_DIR = OUTPUTS / "figures"

# variant -> (test csv, model dir, filename suffix, plot title suffix)
VARIANTS = {
    "whole_slice": (MANIFESTS / "test_holdout.csv", OUTPUTS / "models", "", ""),
    "crop": (MANIFESTS / "test_holdout_crop.csv", OUTPUTS / "models_crop", "_crop", " (crop)"),
    "pathology": (MANIFESTS / "test_holdout_final.csv", OUTPUTS / "models_final", "_final", " (pathology-corrected)"),
    "combined": (MANIFESTS / "test_holdout_combined.csv", OUTPUTS / "models_combined", "_combined", " (combined)"),
    "full": (MANIFESTS / "test_holdout_full.csv", OUTPUTS / "models_full", "_full", " (citra penuh)"),
}


def probs_for_members(predictor, images, members, weighted=False):
    return predictor.predict_batch_probs(images, members=members, weighted=weighted)


def metrics_from_probs(y_true, probs):
    y_pred = probs.argmax(1)
    acc = float(np.mean(y_pred == y_true))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    malignant_idx = CLASS_NAMES.index("Malignant")
    cancer_recall = float(np.mean(y_pred[y_true == malignant_idx] == malignant_idx)) if (y_true == malignant_idx).any() else 0.0
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    benign_idx = CLASS_NAMES.index("Benign")
    benign_recall = float(np.mean(y_pred[y_true == benign_idx] == benign_idx)) if (y_true == benign_idx).any() else float("nan")
    return {"accuracy": acc, "macro_f1": macro_f1, "cancer_recall": cancer_recall,
            "benign_recall": benign_recall, "roc_auc": auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="combined",
                         help="Which experiment stage (Bab IV) to evaluate.")
    parser.add_argument("--no-augment", action="store_true",
                         help="Evaluate the ablation models trained without augmentation.")
    parser.add_argument("--augment-strength", default="medium",
                         choices=["ct", "light", "medium", "heavy"],
                         help="Which augmentation preset the checkpoints were trained with.")
    parser.add_argument("--img-size", type=int, default=224,
                         help="Must match the resolution the models were trained at.")
    args = parser.parse_args()
    test_csv, model_dir, suffix, title_suffix = VARIANTS[args.variant]
    if args.no_augment:
        model_dir = model_dir.with_name(model_dir.name + "_noaug")
        suffix = suffix + "_noaug"
    elif args.augment_strength != "medium":
        model_dir = model_dir.with_name(f"{model_dir.name}_aug{args.augment_strength}")
        suffix = suffix + f"_aug{args.augment_strength}"

    test_df = pd.read_csv(test_csv)
    y_true = test_df["canonical_label"].map({c: i for i, c in enumerate(CLASS_NAMES)}).values
    print(f"Variant: {args.variant} | Held-out test: {len(test_df)} images | "
          f"img_size={args.img_size}", flush=True)

    predictor = EnsemblePredictor(model_dir, img_size=args.img_size)
    print(f"Loaded {len(predictor.member_keys)} models: {predictor.member_keys}", flush=True)

    print("Loading test images into memory...", flush=True)
    images = [Image.open(p) for p in test_df["path"]]

    # per-model (single-fold) results
    single_rows = []
    for key in predictor.member_keys:
        probs = probs_for_members(predictor, images, [key])
        m = metrics_from_probs(y_true, probs)
        single_rows.append({"model": key, **m})
        print(f"  {key}: {m}", flush=True)
    pd.DataFrame(single_rows).to_csv(REPORT_DIR / f"single_model_results{suffix}.csv", index=False)

    # ensemble configurations
    eff_keys = [k for k in predictor.member_keys if k.startswith("efficientnet_b0")]
    res_keys = [k for k in predictor.member_keys if k.startswith("resnet50")]
    configs = {
        "EfficientNet-B0 (5-fold)": eff_keys,
        "ResNet50 (5-fold)": res_keys,
        "Ensemble 10-model (bobot setara)": predictor.member_keys,
        "Ensemble 10-model (bobot tertimbang)": predictor.member_keys,
    }
    ens_rows, ens_probs_cache = [], {}
    for name, members in configs.items():
        weighted = "tertimbang" in name
        probs = probs_for_members(predictor, images, members, weighted=weighted)
        ens_probs_cache[name] = probs
        m = metrics_from_probs(y_true, probs)
        ens_rows.append({"config": name, "n_members": len(members), **m})
        print(f"  {name}: {m}", flush=True)
    ens_df = pd.DataFrame(ens_rows)
    ens_df.to_csv(REPORT_DIR / f"ensemble_results{suffix}.csv", index=False)

    # final ensemble = best macro-F1 among the configs above
    final_name = ens_df.sort_values("macro_f1", ascending=False).iloc[0]["config"]
    final_probs = ens_probs_cache[final_name]
    y_pred_final = final_probs.argmax(1)
    print(f"\nFinal ensemble selected: {final_name}", flush=True)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred_final, labels=[0, 1, 2])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Prediksi"); plt.ylabel("Label Sebenarnya")
    plt.title(f"Confusion Matrix{title_suffix}\n{final_name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"confusion_matrix_ensemble_final{suffix}.png", dpi=150)
    plt.close()

    # per-class metrics
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred_final, labels=[0, 1, 2], zero_division=0)
    per_class = pd.DataFrame({"class": CLASS_NAMES, "support": support, "precision": prec, "recall": rec, "f1": f1})
    per_class.to_csv(REPORT_DIR / f"per_class_metrics_ensemble{suffix}.csv", index=False)
    print(per_class, flush=True)

    # ROC curves (one-vs-rest)
    plt.figure(figsize=(5, 5))
    aucs = {}
    for i, cname in enumerate(CLASS_NAMES):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, final_probs[:, i])
        auc_i = roc_auc_score(y_bin, final_probs[:, i])
        aucs[cname] = auc_i
        plt.plot(fpr, tpr, label=f"{cname} (AUC={auc_i:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"Kurva ROC (One-vs-Rest) - Ensemble Final{title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"roc_curve_ensemble_final{suffix}.png", dpi=150)
    plt.close()

    # misclassified samples
    mis_idx = np.where(y_pred_final != y_true)[0]
    mis_df = test_df.iloc[mis_idx].copy()
    mis_df["true_label"] = [CLASS_NAMES[i] for i in y_true[mis_idx]]
    mis_df["pred_label"] = [CLASS_NAMES[i] for i in y_pred_final[mis_idx]]
    mis_df["pred_confidence"] = final_probs[mis_idx, y_pred_final[mis_idx]]
    mis_df.to_csv(REPORT_DIR / f"misclassified_samples{suffix}.csv", index=False)

    summary = {
        "final_ensemble_config": final_name,
        "final_metrics": metrics_from_probs(y_true, final_probs),
        "per_class_auc": aucs,
        "n_test": len(test_df),
        "n_misclassified": int(len(mis_idx)),
    }
    with open(REPORT_DIR / f"evaluation_summary{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
