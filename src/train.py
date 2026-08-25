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
    from sklearn.metrics import classification_report, confusion_matrix, recall_score

    from dataset import get_dataloaders
    from engine import evaluate
    from train_core import run_training

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    ROOT = SRC_DIR.parent
    DATA_ROOT = ROOT / "dataset_split"
    MODELS_DIR = ROOT / "models"
    REPORTS_DIR = ROOT / "reports"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)

    # v6: training set diperbesar dari 2.012 -> 2.855 dengan menambahkan 80% dari
    # dataset eksternal IQ-OTH/NCCD (data CT asli terverifikasi, bukan duplikat/leakage
    # dari sumber yang sama). B0 (4M param) + regularisasi diperkuat dari v1.
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    MODEL_NAME = "efficientnet_b0"
    IMG_SIZE = 224
    DROP_RATE = 0.3
    DROP_PATH_RATE = 0.2
    LABEL_SMOOTHING = 0.1
    WEIGHT_DECAY = 1.2e-4
    CANCER_RECALL_BOOST = 1.3

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, img_size=IMG_SIZE
    )
    print("Classes:", class_names, flush=True)
    print(
        f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  "
        f"Test: {len(test_loader.dataset)}",
        flush=True,
    )

    with open(MODELS_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f)

    BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
    result = run_training(
        train_loader, val_loader, class_names, device, BEST_MODEL_PATH,
        model_name=MODEL_NAME, drop_rate=DROP_RATE, drop_path_rate=DROP_PATH_RATE,
        label_smoothing=LABEL_SMOOTHING, weight_decay=WEIGHT_DECAY,
        cancer_recall_boost=CANCER_RECALL_BOOST,
    )
    model = result["model"]
    history = result["history"]
    best_val_loss = result["best_val_loss"]
    print(f"Best val_loss selama training: {best_val_loss:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Kurva training
    # ------------------------------------------------------------------
    history_df = pd.DataFrame(history)
    history_df.to_json(REPORTS_DIR / "training_history.json", orient="records", indent=2)

    phase_boundary = (history_df["phase"] == "Fase1-Transfer").sum()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0].axvline(phase_boundary + 0.5, color="gray", linestyle="--", label="fase transisi")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="train_acc")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="val_acc")
    axes[1].axvline(phase_boundary + 0.5, color="gray", linestyle="--", label="fase transisi")
    axes[1].set_title("Akurasi")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].bar(
        history_df["epoch"], history_df["gap"],
        color=history_df["gap_status"].map(
            {"OK": "tab:green", "WARNING": "tab:orange", "OVERFITTING": "tab:red"}
        ),
    )
    axes[2].axhline(0.05, color="orange", linestyle=":", linewidth=1)
    axes[2].axhline(0.15, color="red", linestyle=":", linewidth=1)
    axes[2].set_title("Gap Train-Val Accuracy")
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "training_curves.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Evaluasi test set (model terbaik)
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}", flush=True)

    final_train_loss, final_train_acc, _, _ = evaluate(model, train_loader, criterion, device)
    final_gap = final_train_acc - test_acc
    print(
        f"Train (eval mode) acc: {final_train_acc:.4f}  Test acc: {test_acc:.4f}  Gap: {final_gap:+.4f}",
        flush=True,
    )

    report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
    print(report, flush=True)
    with open(REPORTS_DIR / "test_classification_report.txt", "w") as f:
        f.write(report)

    cancer_idx = class_names.index("cancer")
    cancer_recall = recall_score(test_labels, test_preds, pos_label=cancer_idx)
    print(f"Recall kelas cancer: {cancer_recall:.4f}", flush=True)
    if cancer_recall < 0.90:
        print("PERINGATAN: recall kelas cancer di bawah 0.90 — risiko false negative tinggi.", flush=True)
    else:
        print("Recall kelas cancer memenuhi target minimum (>= 0.90).", flush=True)

    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Ringkasan akhir
    # ------------------------------------------------------------------
    summary = {
        "class_names": class_names,
        "random_seed": RANDOM_SEED,
        "config": {
            "model_name": MODEL_NAME,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "drop_rate": DROP_RATE,
            "drop_path_rate": DROP_PATH_RATE,
            "label_smoothing": LABEL_SMOOTHING,
            "weight_decay": WEIGHT_DECAY,
            "cancer_recall_boost": CANCER_RECALL_BOOST,
            "class_weights": result["class_weights"],
        },
        "epochs_actually_run": len(history_df),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "cancer_recall": float(cancer_recall),
        "train_test_gap": float(final_gap),
        "best_model_path": str(BEST_MODEL_PATH),
    }

    with open(REPORTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("TRAINING SELESAI. Ringkasan disimpan ke reports/training_summary.json", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
