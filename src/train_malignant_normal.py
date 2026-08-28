"""Eksperimen: klasifikasi malignant vs normal (exclude benign), khusus di
IQ-OTH/NCCD, dengan split cluster-safe (lihat src/build_malignant_normal_task.py).
Menguji apakah task yang lebih sempit/mudah ini bisa mencapai akurasi setara
paper-paper yang melaporkan 94-98% di dataset yang sama.
"""

import sys
import json
import random
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import torch
    import torch.nn as nn
    from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_auc_score

    from dataset import get_dataloaders
    from engine import evaluate
    from train_core import run_training

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    ROOT = SRC_DIR.parent
    DATA_ROOT = ROOT / "dataset_split_malignant_vs_normal"
    MODELS_DIR = ROOT / "models"
    REPORTS_DIR = ROOT / "reports"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        DATA_ROOT, batch_size=32, num_workers=2, img_size=224
    )
    print("Classes:", class_names, flush=True)
    print(
        f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  "
        f"Test: {len(test_loader.dataset)}",
        flush=True,
    )

    BEST_MODEL_PATH = MODELS_DIR / "malignant_vs_normal_v9.pth"
    result = run_training(
        train_loader, val_loader, class_names, device, BEST_MODEL_PATH,
        drop_rate=0.3, drop_path_rate=0.2, weight_decay=1.2e-4,
        cancer_recall_boost=1.0,  # tidak perlu boost, task sudah relatif seimbang & lebih mudah
        phase1_epochs=8, phase1_patience=4,
        phase2_epochs=15, phase2_patience=5, warmup_epochs=3,
    )
    model = result["model"]
    history = result["history"]
    best_val_loss = result["best_val_loss"]
    print(f"Best val_loss: {best_val_loss:.4f}", flush=True)

    history_df = pd.DataFrame(history)
    history_df.to_json(REPORTS_DIR / "training_history_v9.json", orient="records", indent=2)

    phase_boundary = (history_df["phase"] == "Fase1-Transfer").sum()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0].axvline(phase_boundary + 0.5, color="gray", linestyle="--")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="train_acc")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="val_acc")
    axes[1].axvline(phase_boundary + 0.5, color="gray", linestyle="--")
    axes[1].set_title("Akurasi")
    axes[1].legend()
    axes[2].bar(
        history_df["epoch"], history_df["gap"],
        color=history_df["gap_status"].map({"OK": "tab:green", "WARNING": "tab:orange", "OVERFITTING": "tab:red"}),
    )
    axes[2].axhline(0.05, color="orange", linestyle=":")
    axes[2].axhline(0.15, color="red", linestyle=":")
    axes[2].set_title("Gap Train-Val Accuracy")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "training_curves_v9.png", dpi=150)
    plt.close(fig)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}", flush=True)

    final_train_loss, final_train_acc, _, _ = evaluate(model, train_loader, criterion, device)
    final_gap = final_train_acc - test_acc
    print(f"Train acc: {final_train_acc:.4f}  Test acc: {test_acc:.4f}  Gap: {final_gap:+.4f}", flush=True)

    report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
    print(report, flush=True)
    with open(REPORTS_DIR / "test_classification_report_v9.txt", "w") as f:
        f.write(report)

    malignant_idx = class_names.index("malignant")
    malignant_recall = recall_score(test_labels, test_preds, pos_label=malignant_idx)
    print(f"Recall kelas malignant: {malignant_recall:.4f}", flush=True)

    # AUC
    import torch.nn.functional as F
    all_probs = []
    all_bin_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            probs = F.softmax(model(images), dim=1)[:, malignant_idx].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_bin_labels.extend((labels.numpy() == malignant_idx).astype(int).tolist())
    test_auc = roc_auc_score(all_bin_labels, all_probs)
    print(f"ROC-AUC: {test_auc:.4f}", flush=True)

    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix — v9 Malignant vs Normal")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix_v9.png", dpi=150)
    plt.close(fig)

    summary = {
        "task": "malignant_vs_normal (IQ-OTH/NCCD only, benign excluded, cluster-safe split)",
        "class_names": class_names,
        "random_seed": RANDOM_SEED,
        "epochs_actually_run": len(history_df),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_roc_auc": float(test_auc),
        "malignant_recall": float(malignant_recall),
        "train_test_gap": float(final_gap),
        "best_model_path": str(BEST_MODEL_PATH),
    }
    with open(REPORTS_DIR / "training_summary_v9.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("TRAINING SELESAI (v9 malignant vs normal).", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
