"""Training-curve figures, built from the real per-epoch histories written
by train_cv.py rather than redrawn by hand. The phase A -> phase B marker
is what makes the fine-tuning step visible in Bab IV.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import CLASS_NAMES

OUT = Path("D:/skripsi/Kodingan/outputs/figures/final")
OUT.mkdir(parents=True, exist_ok=True)
REPORTS = Path("D:/skripsi/Kodingan/outputs/reports")
MANIFESTS = Path("D:/skripsi/Kodingan/outputs/manifests")
OUTPUTS = Path("D:/skripsi/Kodingan/outputs")

# variant -> (model dir, test csv, history prefix, training resolution)
VARIANTS = {
    "combined": (OUTPUTS / "models_combined", MANIFESTS / "test_holdout_combined.csv", "combined", 224),
    "full": (OUTPUTS / "models_full", MANIFESTS / "test_holdout_full.csv", "full", 512),
}
ACTIVE = "full"
MODELS, TEST_CSV, HIST_PREFIX, IMG_SIZE = VARIANTS[ACTIVE]

plt.rcParams["font.family"] = "DejaVu Sans"


def comma_axis(ax, axis="y", decimals=1):
    """Indonesian decimal separator on tick labels, matching the body text."""
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _p: f"{v:.{decimals}f}".replace(".", ","))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


def training_curves(history_file, out_name):
    hist = json.load(open(REPORTS / history_file))
    df = pd.DataFrame(hist)
    df["step"] = range(1, len(df) + 1)
    phase_split = int((df["phase"] == "A-head").sum())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for ax, (train_col, val_col, ylabel) in zip(
        axes, [("train_loss", "loss", "Loss"), ("train_acc", "acc", "Akurasi")]
    ):
        y_train = df[train_col] * (100 if ylabel == "Akurasi" else 1)
        y_val = df[val_col] * (100 if ylabel == "Akurasi" else 1)
        ax.plot(df["step"], y_train, label="Data latih", color="#1f4e9c", linewidth=1.4)
        ax.plot(df["step"], y_val, label="Data validasi", color="#b3202c",
                linewidth=1.4, linestyle="--")
        if 0 < phase_split < len(df):
            ax.axvline(phase_split + 0.5, color="#777777", linestyle=":", linewidth=1.0)
            ymax = max(y_train.max(), y_val.max())
            ymin = min(y_train.min(), y_val.min())
            ax.text(phase_split + 0.8, ymin + (ymax - ymin) * 0.04,
                    "mulai fine-tuning", fontsize=7.5, color="#555555")
        ax.set_xlabel("Epoch (kumulatif fase A + fase B)", fontsize=9)
        ax.set_ylabel(ylabel + (" (%)" if ylabel == "Akurasi" else ""), fontsize=9.5)
        comma_axis(ax, axis="y", decimals=0 if ylabel == "Akurasi" else 1)
        ax.tick_params(labelsize=8.5)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{out_name} done", flush=True)


if __name__ == "__main__":
    training_curves(f"history_{HIST_PREFIX}_resnet50_fold4.json", "gambar_4_1_kurva_resnet_fold4.png")
    print("\nExtra figures written to", OUT, flush=True)
