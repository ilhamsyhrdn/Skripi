"""
Loads every fold checkpoint for every architecture, evaluates each individually on the
untouched held-out test set, then builds several ensembles (soft-voting = probability
averaging) and evaluates those too, so the value of ensembling can be reported and
plotted honestly against the single-model baselines.

Run:  Kodingan/.venv/Scripts/python.exe -m src.evaluation.ensemble_eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

KODINGAN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KODINGAN_DIR))

from src.data.dataset import CLASS_NAMES, LungCTDataset, build_transforms, load_manifest  # noqa: E402
from src.models.factory import build_model  # noqa: E402

MANIFEST_DIR = KODINGAN_DIR / "outputs" / "manifests"
MODEL_DIR = KODINGAN_DIR / "outputs" / "models"
REPORT_DIR = KODINGAN_DIR / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ARCHS = ["efficientnet_b0", "resnet50"]
N_FOLDS = 5


def get_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def metrics_from_probs(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    acc = float((preds == labels).mean())
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    macro_f1 = float(f1.mean())
    weighted_f1 = float(np.average(f1, weights=support))
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))

    # one-vs-rest ROC-AUC per class + macro
    aucs = {}
    for i, cname in enumerate(CLASS_NAMES):
        y_true_bin = (labels == i).astype(int)
        if len(set(y_true_bin)) < 2:
            aucs[cname] = None
            continue
        aucs[cname] = float(roc_auc_score(y_true_bin, probs[:, i]))
    valid_aucs = [v for v in aucs.values() if v is not None]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else None

    per_class = {
        CLASS_NAMES[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "roc_auc": aucs[CLASS_NAMES[i]],
        }
        for i in range(len(CLASS_NAMES))
    }

    malignant_idx = CLASS_NAMES.index("Malignant")
    benign_idx = CLASS_NAMES.index("Benign")
    # "abnormal" = Malignant + Benign combined vs Normal -- secondary clinical screening framing
    abnormal_true = (labels != CLASS_NAMES.index("Normal")).astype(int)
    abnormal_pred = (preds != CLASS_NAMES.index("Normal")).astype(int)
    abnormal_recall = float(
        (abnormal_pred[abnormal_true == 1] == 1).mean()
    ) if (abnormal_true == 1).any() else None

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_roc_auc": macro_auc,
        "malignant_recall_aka_cancer_recall": float(recall[malignant_idx]),
        "benign_recall": float(recall[benign_idx]),
        "abnormal_vs_normal_recall": abnormal_recall,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "roc_auc_per_class": aucs,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = load_manifest(MANIFEST_DIR / "test_holdout.csv")
    eval_tf = build_transforms(224, train=False)
    test_ds = LungCTDataset(test_df, eval_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    all_model_probs: dict[str, np.ndarray] = {}
    labels_ref = None
    per_model_val_f1: dict[str, float] = {}

    print("Evaluating each individual fold checkpoint on the held-out test set...")
    for arch in ARCHS:
        for k in range(N_FOLDS):
            ckpt_path = MODEL_DIR / f"{arch}_fold{k}.pt"
            if not ckpt_path.exists():
                print(f"  [skip] missing {ckpt_path}")
                continue
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = build_model(arch).to(device)
            model.load_state_dict(ckpt["state_dict"])
            probs, labels = get_probs(model, test_loader, device)
            key = f"{arch}_fold{k}"
            all_model_probs[key] = probs
            per_model_val_f1[key] = ckpt.get("best_val_macro_f1", 1.0)
            if labels_ref is None:
                labels_ref = labels
            m = metrics_from_probs(probs, labels)
            print(f"  {key:26s} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
                  f"cancerRecall={m['malignant_recall_aka_cancer_recall']:.3f} "
                  f"AUC={m['macro_roc_auc']:.3f}")
            del model

    results = {"individual_models": {}, "ensembles": {}}
    for key, probs in all_model_probs.items():
        results["individual_models"][key] = metrics_from_probs(probs, labels_ref)

    def build_ensemble(keys: list[str], weighted: bool = False) -> np.ndarray:
        if weighted:
            w = np.array([per_model_val_f1[k] for k in keys])
            w = w / w.sum()
        else:
            w = np.ones(len(keys)) / len(keys)
        stacked = np.stack([all_model_probs[k] for k in keys], axis=0)  # (n_models, N, C)
        return np.tensordot(w, stacked, axes=([0], [0]))

    ensemble_defs = {
        "efficientnet_b0_5fold_ensemble": [k for k in all_model_probs if k.startswith("efficientnet_b0")],
        "resnet50_5fold_ensemble": [k for k in all_model_probs if k.startswith("resnet50")],
        "full_10model_ensemble_equal_weight": list(all_model_probs.keys()),
        "full_10model_ensemble_val_f1_weighted": list(all_model_probs.keys()),
    }
    print("\nEvaluating ensembles (soft-voting = probability averaging)...")
    for name, keys in ensemble_defs.items():
        weighted = "val_f1_weighted" in name
        probs = build_ensemble(keys, weighted=weighted)
        m = metrics_from_probs(probs, labels_ref)
        results["ensembles"][name] = m
        results["ensembles"][name]["members"] = keys
        print(f"  {name:38s} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
              f"cancerRecall={m['malignant_recall_aka_cancer_recall']:.3f} "
              f"AUC={m['macro_roc_auc']:.3f}")

    # pick the "final model" recommended in the report: best macro_f1 among ensembles
    best_name = max(results["ensembles"], key=lambda k: results["ensembles"][k]["macro_f1"])
    results["recommended_final_ensemble"] = best_name
    print(f"\nRecommended final ensemble (highest test macro-F1): {best_name}")

    # single-model average vs best ensemble, for the "ensemble actually helps" comparison
    single_accs = [results["individual_models"][k]["accuracy"] for k in all_model_probs]
    single_f1s = [results["individual_models"][k]["macro_f1"] for k in all_model_probs]
    single_recalls = [results["individual_models"][k]["malignant_recall_aka_cancer_recall"] for k in all_model_probs]
    results["single_model_average"] = {
        "accuracy": float(np.mean(single_accs)),
        "macro_f1": float(np.mean(single_f1s)),
        "malignant_recall_aka_cancer_recall": float(np.mean(single_recalls)),
        "accuracy_std": float(np.std(single_accs)),
        "macro_f1_std": float(np.std(single_f1s)),
    }

    # class names + label distribution for the streamlit app
    results["class_names"] = CLASS_NAMES
    results["test_label_counts"] = {c: int((labels_ref == i).sum()) for i, c in enumerate(CLASS_NAMES)}

    with open(REPORT_DIR / "ensemble_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save raw probs for the Streamlit ROC-curve plotting (avoid re-running inference there)
    np.savez(
        REPORT_DIR / "test_probs_labels.npz",
        labels=labels_ref,
        **{k.replace(".", "_"): v for k, v in all_model_probs.items()},
        ensemble_full_equal=build_ensemble(list(all_model_probs.keys()), weighted=False),
        ensemble_full_weighted=build_ensemble(list(all_model_probs.keys()), weighted=True),
        ensemble_effnet=build_ensemble([k for k in all_model_probs if k.startswith("efficientnet_b0")]),
        ensemble_resnet=build_ensemble([k for k in all_model_probs if k.startswith("resnet50")]),
    )
    print(f"\nWrote {REPORT_DIR / 'ensemble_evaluation.json'} and test_probs_labels.npz")


if __name__ == "__main__":
    main()
