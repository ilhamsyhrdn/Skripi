"""Regenerate every figure used in the thesis, clean and caption-free.

Titles are deliberately NOT baked into the images: in a skripsi the caption
("Gambar 3.1 Diagram Alur Penelitian") is document text placed below the
figure, so a title inside the image would duplicate it.

Outputs go to outputs/figures/final/ and are copied into the manuscript by
build_docx.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve, auc

import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import CLASS_NAMES

VARIANT = os.environ.get("FIG_VARIANT", "full")
OUT = Path("D:/skripsi/Kodingan/outputs/figures/final")
OUT.mkdir(parents=True, exist_ok=True)
MANIFESTS = Path("D:/skripsi/Kodingan/outputs/manifests")
REPORTS = Path("D:/skripsi/Kodingan/outputs/reports")

FONT = "DejaVu Sans"
EDGE = "#1a1a1a"
FILL = "#ffffff"
plt.rcParams["font.family"] = FONT


def koma(value, decimals=2):
    """Indonesian decimal separator, so figures match the body text."""
    return f"{value:.{decimals}f}".replace(".", ",")


def comma_axis(ax, axis="both", decimals=2):
    """Format tick labels with a comma decimal separator."""
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _pos: koma(v, decimals))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


# ---------------------------------------------------------------- flowchart
def _rect(ax, cx, cy, w, h, text, fs=7.5):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, linewidth=1.0,
                            edgecolor=EDGE, facecolor=FILL))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, linespacing=1.35)


def _oval(ax, cx, cy, w, h, text, fs=7.5):
    ax.add_patch(Ellipse((cx, cy), w, h, linewidth=1.0, edgecolor=EDGE, facecolor=FILL))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)


def _para(ax, cx, cy, w, h, text, fs=7.5, skew=0.28):
    pts = [(cx - w / 2 + skew, cy - h / 2), (cx + w / 2 + skew, cy - h / 2),
           (cx + w / 2 - skew, cy + h / 2), (cx - w / 2 - skew, cy + h / 2)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=1.0, edgecolor=EDGE, facecolor=FILL))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, linespacing=1.35)


def _diamond(ax, cx, cy, w, h, text, fs=7.5):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=1.0, edgecolor=EDGE, facecolor=FILL))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, linespacing=1.3)


def _arrow(ax, p1, p2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=9,
                                  color=EDGE, linewidth=0.9,
                                  connectionstyle=f"arc3,rad={rad}"))


def _elbow(ax, pts):
    """Orthogonal polyline with an arrowhead on the final segment."""
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=EDGE, linewidth=0.9, solid_capstyle="butt")
    _arrow(ax, pts[-2], pts[-1])


def flowchart():
    fig, ax = plt.subplots(figsize=(6.0, 13.0))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 23)

    cx = 4.2
    W, H = 4.6, 1.15

    _oval(ax, cx, 22.2, 2.0, 0.85, "Mulai")
    _para(ax, cx, 20.5, W, H, "Akuisisi dataset\n(7 dataset Kaggle, LIDC-IDRI)")
    _rect(ax, cx, 18.7, W, H, "Audit & deduplikasi dataset Kaggle\n(MD5 + perceptual hash)")
    _rect(ax, cx, 16.9, W, H + 0.15, "Pemrosesan LIDC-IDRI\n(parsing XML, windowing HU,\nirisan utuh 512x512)")
    _rect(ax, cx, 15.0, W, H + 0.15, "Koreksi label\n(nearest-neighbor +\ndiagnosis histopatologi TCIA)")
    _rect(ax, cx, 13.1, W, H + 0.15, "Penggabungan dataset &\npembagian data\n(Stratified Group 5-Fold)")
    _rect(ax, cx, 11.2, W, H + 0.15, "Pelatihan transfer learning\n(EfficientNet-B0 & ResNet50,\naugmentasi + fine-tuning)")
    _diamond(ax, cx, 9.0, 3.6, 1.9, "Evaluasi\nmodel")
    _rect(ax, 8.2, 11.2, 2.8, 1.5, "Penyetelan ulang\n(augmentasi, laju\npembelajaran, fine-tuning)", fs=7)
    _rect(ax, cx, 6.9, W, H, "Ensemble model &\npenyesuaian ambang keputusan")
    _rect(ax, cx, 5.2, W, H, "Pembuatan aplikasi\nberbasis web (Streamlit)")
    _diamond(ax, cx, 3.2, 3.6, 1.9, "Black Box\nTesting")
    _rect(ax, 8.2, 5.2, 2.6, 1.0, "Perbaikan kode", fs=7)
    _rect(ax, cx, 1.4, W, H, "Penulisan laporan skripsi")
    _oval(ax, cx, 0.35, 2.0, 0.85, "Selesai")

    # main vertical flow
    chain = [22.2 - 0.43, 20.5 + H / 2, 20.5 - H / 2, 18.7 + H / 2, 18.7 - H / 2,
             16.9 + 0.65, 16.9 - 0.65, 15.0 + 0.65, 15.0 - 0.65, 13.1 + 0.65,
             13.1 - 0.65, 11.2 + 0.65, 11.2 - 0.65, 9.0 + 0.95]
    for i in range(0, len(chain) - 1, 2):
        _arrow(ax, (cx, chain[i]), (cx, chain[i + 1]))
    _arrow(ax, (cx, 9.0 - 0.95), (cx, 6.9 + H / 2))
    _arrow(ax, (cx, 6.9 - H / 2), (cx, 5.2 + H / 2))
    _arrow(ax, (cx, 5.2 - H / 2), (cx, 3.2 + 0.95))
    _arrow(ax, (cx, 3.2 - 0.95), (cx, 1.4 + H / 2))
    _arrow(ax, (cx, 1.4 - H / 2), (cx, 0.35 + 0.43))

    # feedback loop: evaluation -> scenario fix -> training
    _elbow(ax, [(cx + 1.8, 9.0), (8.2, 9.0), (8.2, 11.2 - 0.75)])
    _elbow(ax, [(8.2, 11.2 + 0.75), (8.2, 12.1), (cx + W / 2, 12.1), (cx + W / 2 - 0.4, 12.1)])
    ax.text(cx + 2.0, 9.25, "kurang baik", fontsize=6.5, ha="left")
    ax.text(cx + 0.15, 8.0, "baik", fontsize=6.5, ha="left")

    # feedback loop: black box testing -> code fix -> web app
    _elbow(ax, [(cx + 1.8, 3.2), (8.2, 3.2), (8.2, 5.2 - 0.5)])
    _elbow(ax, [(8.2, 5.2 + 0.5), (8.2, 5.95), (cx + W / 2, 5.95), (cx + W / 2 - 0.4, 5.95)])
    ax.text(cx + 2.0, 3.45, "tidak lulus", fontsize=6.5, ha="left")
    ax.text(cx + 0.15, 2.25, "lulus", fontsize=6.5, ha="left")

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "gambar_3_1_alur_penelitian.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("flowchart done", flush=True)


# ------------------------------------------------------------ use case diagram
def use_case():
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)

    # system boundary
    ax.add_patch(Rectangle((3.1, 0.3), 7.5, 7.4, linewidth=1.0, edgecolor=EDGE,
                            facecolor="none"))
    ax.text(6.85, 7.4, "Sistem Klasifikasi Kanker Paru-Paru", ha="center",
            fontsize=8.5)

    ax_x, ax_y = 1.4, 4.0
    ax.add_patch(Ellipse((ax_x, ax_y + 1.15), 0.5, 0.5, fill=False, linewidth=1.0,
                          edgecolor=EDGE))
    ax.plot([ax_x, ax_x], [ax_y + 0.9, ax_y - 0.15], color=EDGE, linewidth=1.0)
    ax.plot([ax_x - 0.45, ax_x + 0.45], [ax_y + 0.55, ax_y + 0.55], color=EDGE, linewidth=1.0)
    ax.plot([ax_x, ax_x - 0.38], [ax_y - 0.15, ax_y - 0.85], color=EDGE, linewidth=1.0)
    ax.plot([ax_x, ax_x + 0.38], [ax_y - 0.15, ax_y - 0.85], color=EDGE, linewidth=1.0)
    ax.text(ax_x, ax_y - 1.2, "Pengguna", ha="center", fontsize=8.5)

    ucs = {
        "upload": ("Mengunggah\ncitra CT", 5.1, 6.5),
        "preview": ("Melihat pratinjau\ncitra", 5.1, 4.9),
        "classify": ("Melakukan\nklasifikasi", 5.1, 3.3),
        "threshold": ("Mengatur ambang\nkeputusan", 8.7, 4.1),
        "result": ("Melihat hasil &\nconfidence score", 5.1, 1.7),
        "members": ("Melihat probabilitas\ntiap anggota ensemble", 8.7, 2.3),
    }
    pos = {}
    for key, (label, x, y) in ucs.items():
        ax.add_patch(Ellipse((x, y), 2.9, 1.15, linewidth=1.0, edgecolor=EDGE,
                              facecolor=FILL))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5, linespacing=1.3)
        pos[key] = (x, y)

    for key in ["upload", "preview", "classify", "result"]:
        x, y = pos[key]
        ax.plot([ax_x + 0.45, x - 1.45], [ax_y, y], color=EDGE, linewidth=0.8)

    def dashed(a, b, label):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.plot([x1 + 1.35, x2 - 1.35], [y1, y2], linestyle=(0, (4, 3)),
                color="#555555", linewidth=0.8)
        ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2 + 0.16, label, fontsize=6.5,
                style="italic", color="#555555", ha="center")

    dashed("classify", "threshold", "<<extend>>")
    dashed("result", "members", "<<extend>>")

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "gambar_3_5_use_case.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("use case done", flush=True)


# --------------------------------------------------------------- sample grids
def sample_grid(df, out_name, n=3, classes=("Malignant", "Benign", "Normal")):
    fig, axes = plt.subplots(len(classes), n, figsize=(n * 1.9, len(classes) * 2.05))
    for r, cls in enumerate(classes):
        sub = df[df["canonical_label"] == cls].head(n)
        for c in range(n):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.6); s.set_color("#888888")
            if c < len(sub):
                with Image.open(sub.iloc[c]["path"]) as im:
                    ax.imshow(im.convert("L"), cmap="gray")
            if c == 0:
                ax.set_ylabel(cls, fontsize=9.5, labelpad=6)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{out_name} done", flush=True)


def samples():
    kaggle = pd.read_csv(MANIFESTS / "kaggle_canonical_pool.csv")
    sample_grid(kaggle[kaggle["source_dataset"] == "The IQ-OTHNCCD lung cancer dataset ( Hamdalla F. Al-Yasriy )"],
                "gambar_3_2_sampel_kaggle_alyasriy.png")
    sample_grid(kaggle[kaggle["source_dataset"] == "CT Scan Images for Lung Cancer (Dishan rathi20)"],
                "gambar_3_3_sampel_kaggle_rathi.png")
    lidc = pd.read_csv(MANIFESTS / "lidc_canonical_pool_full.csv")
    sample_grid(lidc, "gambar_3_4_sampel_lidc.png")


# ---------------------------------------------------------- bab ii schematics
def _chain(ax, labels, y=0.5, w=1.75, h=1.15, gap=0.42, fs=7.5):
    x = 0.25
    centers = []
    for text in labels:
        ax.add_patch(Rectangle((x, y), w, h, linewidth=1.0, edgecolor=EDGE, facecolor=FILL))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                linespacing=1.35)
        centers.append((x, x + w))
        x += w + gap
    for i in range(len(centers) - 1):
        _arrow(ax, (centers[i][1], y + h / 2), (centers[i + 1][0], y + h / 2))
    return x, centers


def cnn_diagram():
    fig, ax = plt.subplots(figsize=(9.2, 2.1))
    ax.axis("off")
    labels = ["Citra masukan\n224 x 224", "Lapisan\nkonvolusi", "Lapisan\npooling",
              "Lapisan konvolusi\n& pooling\n(berulang)", "Fully\nconnected",
              "Keluaran\n3 kelas"]
    xmax, _ = _chain(ax, labels, y=0.35, w=1.62, h=1.25, gap=0.36, fs=7)
    ax.plot([0.25, 0.25 + 1.62 * 4 + 0.36 * 3], [0.18, 0.18], color="#666666", linewidth=0.8)
    ax.text((0.25 + 1.62 * 4 + 0.36 * 3) / 2, 0.02, "ekstraksi fitur", fontsize=7,
            ha="center", color="#444444")
    ax.plot([0.25 + 1.62 * 4 + 0.36 * 4, xmax - 0.42], [0.18, 0.18], color="#666666", linewidth=0.8)
    ax.text((0.25 + 1.62 * 4 + 0.36 * 4 + xmax - 0.42) / 2, 0.02, "klasifikasi",
            fontsize=7, ha="center", color="#444444")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.12, 1.75)
    fig.tight_layout(pad=0.25)
    fig.savefig(OUT / "gambar_2_1_arsitektur_cnn.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("cnn diagram done", flush=True)


def block_diagram(labels, skip_label, out_name):
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    ax.axis("off")
    xmax, centers = _chain(ax, labels, y=0.3, w=1.62, h=1.2, gap=0.38, fs=7)
    x_first = (centers[0][0] + centers[0][1]) / 2
    x_last = (centers[-1][0] + centers[-1][1]) / 2
    ax.annotate("", xy=(x_last, 1.55), xytext=(x_first, 1.55),
                arrowprops=dict(arrowstyle="-|>", color="#555555", linewidth=0.9,
                                 connectionstyle="arc3,rad=-0.22"))
    ax.text((x_first + x_last) / 2, 2.12, skip_label, ha="center", fontsize=7,
            color="#555555")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 2.45)
    fig.tight_layout(pad=0.25)
    fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{out_name} done", flush=True)


# ------------------------------------------------------------- bab iv results
def _final_pred():
    probs = np.load(REPORTS / f"stacking_probs_{VARIANT}.npy")
    test_df = pd.read_csv(MANIFESTS / f"test_holdout_{VARIANT}.csv")
    l2i = {c: i for i, c in enumerate(CLASS_NAMES)}
    y_true = test_df["canonical_label"].map(l2i).values
    mal = CLASS_NAMES.index("Malignant")
    y_pred = np.where(probs[:, mal] >= 0.5, mal, probs.argmax(1))
    return y_true, y_pred, probs


def confusion_fig():
    y_true, y_pred, _ = _final_pred()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks(range(3), CLASS_NAMES, fontsize=9)
    ax.set_yticks(range(3), CLASS_NAMES, fontsize=9, rotation=90, va="center")
    ax.set_xlabel("Prediksi model", fontsize=9.5, labelpad=6)
    ax.set_ylabel("Label sebenarnya", fontsize=9.5, labelpad=6)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "#111111")
    cb = fig.colorbar(im, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "gambar_4_5_confusion_matrix.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("confusion matrix done", flush=True)


def roc_fig():
    y_true, _, probs = _final_pred()
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    styles = {"Benign": ("#2a7f3e", "--"), "Malignant": ("#b3202c", "-"),
              "Normal": ("#1f4e9c", "-.")}
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), probs[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC = {koma(auc(fpr, tpr), 3)})",
                color=styles[cls][0], linestyle=styles[cls][1], linewidth=1.4)
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.9, linestyle=":")
    ax.set_xlabel("False Positive Rate", fontsize=9.5)
    ax.set_ylabel("True Positive Rate", fontsize=9.5)
    comma_axis(ax, axis="both", decimals=1)
    ax.tick_params(labelsize=8.5)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.95)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "gambar_4_6_kurva_roc.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("roc done", flush=True)


def threshold_fig():
    df = pd.read_csv(REPORTS / f"stacking_threshold_sweep_{VARIANT}.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(df["threshold"], df["accuracy"] * 100, marker="o", markersize=4.5,
            linewidth=1.4, color="#1f4e9c", label="Akurasi")
    ax.plot(df["threshold"], df["cancer_recall"] * 100, marker="s", markersize=4.5,
            linewidth=1.4, color="#b3202c", label="Cancer recall")
    ax.plot(df["threshold"], df["cancer_precision"] * 100, marker="^", markersize=4.5,
            linewidth=1.4, color="#2a7f3e", label="Cancer precision")

    ymin, ymax = 72, 100
    ax.set_ylim(ymin, ymax)
    ax.axvline(0.50, color="#666666", linestyle="--", linewidth=1.0)
    # annotation anchored at the TOP of the plot area, clear of the tick labels
    ax.annotate("ambang terpilih (0,50)", xy=(0.50, 97.2), xytext=(0.455, 97.2),
                fontsize=8, color="#333333", va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          edgecolor="#999999", linewidth=0.7),
                arrowprops=dict(arrowstyle="-|>", color="#666666", linewidth=0.8))

    ax.set_xlabel("Ambang keputusan kelas Malignant", fontsize=10)
    ax.set_ylabel("Persentase (%)", fontsize=10)
    ax.set_xticks(list(df["threshold"]))
    comma_axis(ax, axis="x", decimals=2)
    comma_axis(ax, axis="y", decimals=0)
    ax.tick_params(labelsize=9)
    ax.invert_xaxis()
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.95)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "gambar_4_7_sweep_ambang.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("threshold sweep done", flush=True)


def _grouped_bars(ax, labels, series, ylim=(0, 105), decimals=1, fs_label=8):
    """Grouped bars with value labels -- the shared shape of every ablation figure."""
    x = np.arange(len(labels))
    w = 0.8 / len(series)
    colors = ["#1f4e9c", "#b3202c", "#2a7f3e"]
    for i, (name, vals) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * w
        rects = ax.bar(x + off, vals, w, color=colors[i % len(colors)], label=name)
        for rect, v in zip(rects, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + (ylim[1] - ylim[0]) * 0.012,
                    koma(v, decimals), ha="center", fontsize=7)
    ax.set_xticks(x, labels, fontsize=fs_label)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)


def finetuning_fig():
    """Kontribusi fine-tuning per model: garis dari fase A ke fase B.

    Bar berdampingan tidak terbaca di sini karena selisihnya kecil dan
    label angkanya saling bertumpuk; bentuk dumbbell menampilkan arah
    dan besar perubahan tiap model sekaligus.
    """
    df = pd.read_csv(REPORTS / f"ablasi_finetuning_{VARIANT}.csv").iloc[::-1]
    labels = [m.replace("efficientnet_b0_fold", "EfficientNet-B0 fold ")
               .replace("resnet50_fold", "ResNet50 fold ") for m in df["model"]]
    y = np.arange(len(df))
    a, b = df["f1_fase_A"].values, df["f1_fase_B"].values

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for i in range(len(df)):
        naik = b[i] >= a[i]
        ax.annotate("", xy=(b[i], y[i]), xytext=(a[i], y[i]),
                    arrowprops=dict(arrowstyle="-|>", linewidth=1.5,
                                    color="#2a7f3e" if naik else "#b3202c"))
    ax.scatter(a, y, s=42, color="#1f4e9c", zorder=3, label="Fase A: feature extraction")
    ax.scatter(b, y, s=42, color="#b3202c", zorder=3, marker="D",
               label="Fase B: setelah fine-tuning")
    for i in range(len(df)):
        ax.text(max(a[i], b[i]) + 0.004, y[i], koma(b[i] - a[i], 4).replace("-", "−"),
                va="center", fontsize=7.5,
                color="#2a7f3e" if b[i] >= a[i] else "#b3202c")

    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_xlabel("Macro-F1 validasi terbaik", fontsize=9.5)
    ax.set_xlim(0.60, 0.775)
    comma_axis(ax, axis="x", decimals=2)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8.5, ncol=2, frameon=False,
              loc="lower left", bbox_to_anchor=(0, 1.005))
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "gambar_4_2_kontribusi_finetuning.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("kontribusi fine-tuning done", flush=True)


def augmentation_fig():
    """Kontribusi augmentasi: tiga kekuatan, akurasi uji vs jurang overfitting.

    Satu sumbu saja menyesatkan di sini. Augmentasi jelas menekan overfitting
    (jurang turun rapi 10,0 -> 5,3 -> 0,7 poin) tanpa menaikkan akurasi uji,
    jadi kedua besaran ditampilkan berdampingan.
    """
    src = REPORTS / f"ablasi_augmentasi_{VARIANT}.csv"
    if not src.exists():
        print("kontribusi augmentasi DILEWATI (ablasi belum selesai)", flush=True)
        return
    df = pd.read_csv(src).set_index("konfigurasi").loc[["tanpa", "ringan_ct", "penuh"]]
    label = ["Tanpa\naugmentasi", "Augmentasi\nringan (CT)", "Augmentasi\npenuh"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    _grouped_bars(axes[0], label,
                  [("Akurasi", list(df["accuracy"] * 100)),
                   ("Cancer recall", list(df["cancer_recall"] * 100))],
                  ylim=(0, 105))
    axes[0].set_ylabel("Persentase (%)", fontsize=9.5)
    comma_axis(axes[0], axis="y", decimals=0)
    axes[0].legend(fontsize=8.5, ncol=2, frameon=False, loc="lower left",
                   bbox_to_anchor=(0, 1.005))

    x = np.arange(len(df))
    axes[1].bar(x, df["jurang_overfitting"] * 100, 0.5, color="#b3202c")
    for i, v in enumerate(df["jurang_overfitting"] * 100):
        axes[1].text(i, v + 0.25, koma(v, 2), ha="center", fontsize=8)
    axes[1].set_xticks(x, label, fontsize=8)
    axes[1].set_ylim(0, 12)
    axes[1].set_ylabel("Selisih akurasi latih − validasi (poin)", fontsize=9.5)
    comma_axis(axes[1], axis="y", decimals=0)
    axes[1].tick_params(axis="y", labelsize=8.5)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[1].set_title("Jurang overfitting", fontsize=9.5, pad=8)

    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / "gambar_4_3_kontribusi_augmentasi.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("kontribusi augmentasi done", flush=True)


def ensemble_fig():
    """Kontribusi ensemble: model tunggal -> soft-voting -> stacking, pada data uji."""
    single = pd.read_csv(REPORTS / f"single_model_results_{VARIANT}.csv")
    ens = pd.read_csv(REPORTS / f"ensemble_results_{VARIANT}.csv")
    sweep = pd.read_csv(REPORTS / f"stacking_threshold_sweep_{VARIANT}.csv")
    stack = sweep[sweep["threshold"] == 0.50].iloc[0]
    tert = ens[ens["config"].str.contains("tertimbang")].iloc[0]
    setara = ens[ens["config"].str.contains("setara")].iloc[0]

    rows = [("Model tunggal\n(rata-rata 10)", single["accuracy"].mean(),
             single["cancer_recall"].mean()),
            ("Model tunggal\nterbaik", single["accuracy"].max(),
             single.loc[single["accuracy"].idxmax(), "cancer_recall"]),
            ("Soft-voting\n(bobot setara)", setara["accuracy"], setara["cancer_recall"]),
            ("Soft-voting\n(tertimbang)", tert["accuracy"], tert["cancer_recall"]),
            ("Ensemble\nstacking", stack["accuracy"], stack["cancer_recall"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    _grouped_bars(ax, [r[0] for r in rows],
                  [("Akurasi", [r[1] * 100 for r in rows]),
                   ("Cancer recall", [r[2] * 100 for r in rows])])
    ax.set_ylabel("Persentase (%)", fontsize=9.5)
    comma_axis(ax, axis="y", decimals=0)
    ax.legend(fontsize=8.5, ncol=2, frameon=False,
              loc="lower left", bbox_to_anchor=(0, 1.005))
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "gambar_4_4_kontribusi_ensemble.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("kontribusi ensemble done", flush=True)


def per_source_fig():
    """Akurasi dipecah per sumber data, supaya angka gabungan tidak dibaca berlebihan."""
    y_true, y_pred, _ = _final_pred()
    test_df = pd.read_csv(MANIFESTS / f"test_holdout_{VARIANT}.csv")
    is_lidc = test_df["source_dataset"].astype(str).str.upper().str.contains("LIDC").values
    mal = CLASS_NAMES.index("Malignant")

    def stats(m):
        acc = (y_true[m] == y_pred[m]).mean() * 100
        km = m & (y_true == mal)
        return acc, (y_pred[km] == mal).mean() * 100

    groups = [("Kaggle", ~is_lidc), ("LIDC-IDRI", is_lidc),
              ("Gabungan\n(seluruh data uji)", np.ones(len(y_true), bool))]
    vals = [stats(m) for _, m in groups]
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    _grouped_bars(ax, [f"{n}\n(n = {m.sum()})" for n, m in groups],
                  [("Akurasi", [v[0] for v in vals]),
                   ("Cancer recall", [v[1] for v in vals])])
    ax.set_ylabel("Persentase (%)", fontsize=9.5)
    comma_axis(ax, axis="y", decimals=0)
    ax.legend(fontsize=8.5, ncol=2, frameon=False,
              loc="lower left", bbox_to_anchor=(0, 1.005))
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "gambar_4_8_performa_per_sumber.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("performa per sumber done", flush=True)


if __name__ == "__main__":
    cnn_diagram()
    block_diagram(["Masukan\nH x W x C", "Konvolusi 1x1\n(ekspansi)",
                    "Konvolusi\ndepthwise 3x3", "Squeeze-and-\nExcitation",
                    "Konvolusi 1x1\n(proyeksi)", "Keluaran"],
                   "skip connection (bila dimensi sesuai)",
                   "gambar_2_2_blok_mbconv.png")
    block_diagram(["Masukan x", "Konvolusi 1x1\n(reduksi)", "Konvolusi 3x3",
                    "Konvolusi 1x1\n(ekspansi)", "Keluaran\nF(x) + x"],
                   "skip connection: F(x) + x",
                   "gambar_2_3_blok_residual.png")
    flowchart()
    samples()
    use_case()
    confusion_fig()
    roc_fig()
    threshold_fig()
    finetuning_fig()
    augmentation_fig()
    ensemble_fig()
    per_source_fig()
    print("\nAll figures written to", OUT, flush=True)
