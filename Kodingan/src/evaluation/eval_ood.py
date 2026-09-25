"""Evaluate the OOD detector on its held-out test split (165 Lung CT + 165
Bukan Lung CT) and save confusion matrix + metrics for Bab IV."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.ood_dataset import OODDataset, build_ood_transforms, OOD_CLASS_NAMES
from models.factory import build_model

MANIFEST = Path("D:/skripsi/Kodingan/outputs/manifests/ood_manifest.csv")
MODEL_PATH = Path("D:/skripsi/Kodingan/outputs/models/ood_detector.pt")
FIG_DIR = Path("D:/skripsi/Kodingan/outputs/figures")
REPORT_DIR = Path("D:/skripsi/Kodingan/outputs/reports")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(MANIFEST)
    test_df = df[df.split == "test"]
    ds = OODDataset(test_df, build_ood_transforms(train=False))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    model = build_model("efficientnet_b0", n_classes=2).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = float(np.mean(y_true == y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=OOD_CLASS_NAMES, yticklabels=OOD_CLASS_NAMES)
    plt.xlabel("Prediksi"); plt.ylabel("Label Sebenarnya")
    plt.title("Confusion Matrix - Modul Validasi Input")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ood_confusion_matrix.png", dpi=150)
    plt.close()

    summary = {
        "accuracy": acc, "n_test": int(len(test_df)),
        "per_class": {OOD_CLASS_NAMES[i]: {"precision": prec[i], "recall": rec[i], "f1": f1[i], "support": int(support[i])} for i in range(2)},
    }
    with open(REPORT_DIR / "ood_evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
