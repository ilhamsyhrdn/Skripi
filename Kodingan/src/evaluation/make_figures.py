"""Generates static PNG figures (for embedding directly in the written skripsi document,
Bab 4 Hasil dan Pembahasan) from the same artifacts the Streamlit dashboard reads.

Run:  Kodingan/.venv/Scripts/python.exe -m src.evaluation.make_figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

KODINGAN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KODINGAN_DIR))

from src.data.dataset import CLASS_NAMES  # noqa: E402

MANIFEST_DIR = KODINGAN_DIR / "outputs" / "manifests"
REPORT_DIR = KODINGAN_DIR / "outputs" / "reports"
FIG_DIR = KODINGAN_DIR / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
CLASS_COLORS = {"Benign": "#f2b134", "Malignant": "#e63946", "Normal": "#2a9d8f"}


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def fig_dataset_dedup_impact():
    audit = load_json(MANIFEST_DIR / "audit_summary.json")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vals = [audit["total_files_found"], audit["canonical_pool_size"]]
    bars = ax.bar(["Total file mentah\n(7 dataset Kaggle)", "Pool kanonik\n(setelah deduplikasi)"],
                   vals, color=["#a8a8a8", "#2a9d8f"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Jumlah citra")
    ax.set_title("Dampak Audit Deduplikasi terhadap Ukuran Dataset")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_dataset_dedup_impact.png", dpi=150)
    plt.close(fig)


def fig_class_distribution():
    canonical = pd.read_csv(MANIFEST_DIR / "canonical_pool.csv")
    counts = canonical["canonical_label"].value_counts().reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(counts.index, counts.values, color=[CLASS_COLORS[c] for c in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Jumlah citra unik")
    ax.set_title("Distribusi Kelas pada Pool Kanonik (894 citra)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_class_distribution.png", dpi=150)
    plt.close(fig)


def fig_model_comparison():
    ens_eval = load_json(REPORT_DIR / "ensemble_evaluation.json")
    rows = []
    for key, m in ens_eval["individual_models"].items():
        rows.append({"model": key, "type": "single", "accuracy": m["accuracy"],
                     "macro_f1": m["macro_f1"], "cancer_recall": m["malignant_recall_aka_cancer_recall"]})
    for key, m in ens_eval["ensembles"].items():
        rows.append({"model": key, "type": "ensemble", "accuracy": m["accuracy"],
                     "macro_f1": m["macro_f1"], "cancer_recall": m["malignant_recall_aka_cancer_recall"]})
    df = pd.DataFrame(rows).sort_values("macro_f1")

    fig, axes = plt.subplots(1, 3, figsize=(16, max(6, 0.35 * len(df))), sharey=True)
    for ax, metric, title in zip(
        axes, ["accuracy", "macro_f1", "cancer_recall"],
        ["Akurasi", "Macro-F1", "Cancer Recall (Malignant)"],
    ):
        colors = ["#2a9d8f" if t == "ensemble" else "#a8a8a8" for t in df["type"]]
        ax.barh(df["model"], df[metric], color=colors)
        ax.set_title(title)
        ax.set_xlim(0, 1)
    fig.suptitle("Perbandingan Model Tunggal vs Ensemble (Held-out Test Set)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_model_comparison.png", dpi=150)
    plt.close(fig)


def fig_confusion_matrix_and_roc():
    ens_eval = load_json(REPORT_DIR / "ensemble_evaluation.json")
    best_name = ens_eval["recommended_final_ensemble"]
    cm = np.array(ens_eval["ensembles"][best_name]["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    ax.set_title(f"Confusion Matrix -- {best_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_confusion_matrix.png", dpi=150)
    plt.close(fig)

    npz = np.load(REPORT_DIR / "test_probs_labels.npz")
    key_map = {
        "full_10model_ensemble_equal_weight": "ensemble_full_equal",
        "full_10model_ensemble_val_f1_weighted": "ensemble_full_weighted",
        "efficientnet_b0_5fold_ensemble": "ensemble_effnet",
        "resnet50_5fold_ensemble": "ensemble_resnet",
    }
    probs = npz[key_map.get(best_name, "ensemble_full_equal")]
    labels = npz["labels"]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    for i, cname in enumerate(CLASS_NAMES):
        y_true = (labels == i).astype(int)
        if len(set(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs[:, i])
        auc_val = ens_eval["ensembles"][best_name]["roc_auc_per_class"].get(cname)
        ax.plot(fpr, tpr, label=f"{cname} (AUC={auc_val:.3f})", color=CLASS_COLORS[cname])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Kurva ROC (one-vs-rest) -- {best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_roc_curve.png", dpi=150)
    plt.close(fig)


def fig_training_curves_best_fold():
    # pick the effnet fold with the highest val_macro_f1 as the representative curve
    best_fold, best_f1 = None, -1
    for k in range(5):
        h = load_json(REPORT_DIR / f"efficientnet_b0_fold{k}_history.json")
        if h["best_val_macro_f1"] > best_f1:
            best_f1, best_fold, hist = h["best_val_macro_f1"], k, h

    hdf = pd.DataFrame(hist["history"])
    hdf["epoch"] = range(1, len(hdf) + 1)
    phase_change = hdf[hdf["phase"] == "B-finetune"]["epoch"].min()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(hdf["epoch"], hdf["train_loss"], label="Train loss")
    axes[0].plot(hdf["epoch"], hdf["val_loss"], label="Val loss")
    axes[0].axvline(phase_change - 0.5, color="gray", linestyle=":", label="Mulai fine-tuning")
    axes[0].set_title(f"Loss -- EfficientNet-B0 fold {best_fold} (representatif)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(hdf["epoch"], hdf["train_acc"], label="Train acc")
    axes[1].plot(hdf["epoch"], hdf["val_acc"], label="Val acc")
    axes[1].plot(hdf["epoch"], hdf["val_macro_f1"], label="Val macro-F1")
    axes[1].plot(hdf["epoch"], hdf["val_malignant_recall"], label="Val cancer recall")
    axes[1].axvline(phase_change - 0.5, color="gray", linestyle=":")
    axes[1].set_title(f"Accuracy / F1 / Recall -- EfficientNet-B0 fold {best_fold}")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_training_curves_best_fold.png", dpi=150)
    plt.close(fig)


def fig_naive_vs_honest():
    naive_path = REPORT_DIR / "naive_leaky_baseline.json"
    if not naive_path.exists():
        return
    naive = load_json(naive_path)
    ens_eval = load_json(REPORT_DIR / "ensemble_evaluation.json")
    best = ens_eval["ensembles"][ens_eval["recommended_final_ensemble"]]

    metrics = ["Akurasi", "Macro-F1", "Cancer Recall"]
    naive_vals = [naive["test_accuracy"], naive["test_macro_f1"], naive["test_malignant_recall"]]
    honest_vals = [best["accuracy"], best["macro_f1"], best["malignant_recall_aka_cancer_recall"]]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, naive_vals, w, label="Baseline naif (bocor)", color="#e63946")
    ax.bar(x + w / 2, honest_vals, w, label="Pipeline final (bersih)", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Dampak Data Leakage terhadap Estimasi Performa")
    ax.legend()
    for i, (n, h) in enumerate(zip(naive_vals, honest_vals)):
        ax.text(i - w / 2, n, f"{n:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_naive_vs_honest.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_dataset_dedup_impact()
    fig_class_distribution()
    fig_model_comparison()
    fig_confusion_matrix_and_roc()
    fig_training_curves_best_fold()
    fig_naive_vs_honest()
    print(f"Figures written to {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(" -", p.name)
