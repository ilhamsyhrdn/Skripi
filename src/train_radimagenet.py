"""Eksperimen v8: ganti backbone dari EfficientNet-B0 (pretrained ImageNet) ke
ResNet50 pretrained RadImageNet (citra medis, bukan foto natural).

Paket `radimagenet-models` yang dipakai adalah port PyTorch TIDAK RESMI dari
bobot Keras RadImageNet asli, jadi preprocessing/normalisasi persis yang
dipakai saat pretraining tidak terdokumentasi. Digunakan normalisasi ImageNet
standar (asumsi paling mungkin benar, karena arsitektur dasarnya juga
diinisialisasi dari ImageNet sebelum dilanjutkan pretraining di RadImageNet).
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
    from sklearn.metrics import classification_report, confusion_matrix, recall_score

    from dataset import get_dataloaders
    from model import build_radimagenet_model
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    BATCH_SIZE = 32
    NUM_WORKERS = 2
    IMG_SIZE = 224
    DROP_RATE = 0.4  # lebih tinggi dari v6 (0.3) karena backbone 6x lebih besar (23.5M vs 4M param)
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

    weights_path = MODELS_DIR / "RadImageNet-ResNet50_notop.pth"
    model = build_radimagenet_model(
        num_classes=len(class_names), drop_rate=DROP_RATE, weights_path=weights_path
    )
    print(f"Total parameter: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Epoch dikurangi dari default (8+18) karena ResNet50 ~10x lebih lambat per
    # epoch (~150-170s) dibanding EfficientNet-B0 (~14-17s) -- budget waktu total
    # tetap dijaga wajar (~35-40 menit) untuk eksperimen dengan payoff yang belum
    # pasti melebihi ensemble v5+v6 yang sudah ada.
    BEST_MODEL_PATH = MODELS_DIR / "radimagenet_resnet50_v8.pth"
    result = run_training(
        train_loader, val_loader, class_names, device, BEST_MODEL_PATH,
        drop_rate=DROP_RATE, weight_decay=WEIGHT_DECAY, cancer_recall_boost=CANCER_RECALL_BOOST,
        model=model,
        phase1_epochs=5, phase1_patience=3,
        phase2_epochs=10, phase2_patience=4, warmup_epochs=2,
    )
    model = result["model"]
    history = result["history"]
    best_val_loss = result["best_val_loss"]
    print(f"Best val_loss selama training: {best_val_loss:.4f}", flush=True)

    history_df = pd.DataFrame(history)
    history_df.to_json(REPORTS_DIR / "training_history_v8.json", orient="records", indent=2)

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
    plt.savefig(REPORTS_DIR / "training_curves_v8.png", dpi=150)
    plt.close(fig)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}", flush=True)

    final_train_loss, final_train_acc, _, _ = evaluate(model, train_loader, criterion, device)
    final_gap = final_train_acc - test_acc
    print(f"Train acc: {final_train_acc:.4f}  Test acc: {test_acc:.4f}  Gap: {final_gap:+.4f}", flush=True)

    report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
    print(report, flush=True)
    with open(REPORTS_DIR / "test_classification_report_v8.txt", "w") as f:
        f.write(report)

    cancer_idx = class_names.index("cancer")
    cancer_recall = recall_score(test_labels, test_preds, pos_label=cancer_idx)
    print(f"Recall kelas cancer: {cancer_recall:.4f}", flush=True)

    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix — v8 RadImageNet ResNet50")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix_v8.png", dpi=150)
    plt.close(fig)

    summary = {
        "class_names": class_names,
        "random_seed": RANDOM_SEED,
        "config": {
            "model_name": "radimagenet_resnet50",
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "drop_rate": DROP_RATE,
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
    with open(REPORTS_DIR / "training_summary_v8.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("TRAINING SELESAI (v8 RadImageNet ResNet50).", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
