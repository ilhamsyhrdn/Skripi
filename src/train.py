import sys
import time
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
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    from sklearn.metrics import classification_report, confusion_matrix, recall_score

    from dataset import get_dataloaders, compute_class_weights
    from model import build_model, freeze_backbone, unfreeze_all, count_trainable_params
    from engine import train_one_epoch, evaluate

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

    # ------------------------------------------------------------------
    # 1. Data loader & class weights
    # ------------------------------------------------------------------
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    # v5: dataset bersih (2.012 train, turun dari 2.576) terlalu kecil untuk B2 (7.7M param)
    # -> v4 overfitting (gap 20.4%). Balik ke B0 (4M param) + regularisasi lebih kuat.
    MODEL_NAME = "efficientnet_b0"
    IMG_SIZE = 224

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, img_size=IMG_SIZE
    )
    print("Classes:", class_names, flush=True)
    print(
        f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  "
        f"Test: {len(test_loader.dataset)}",
        flush=True,
    )

    class_weights = compute_class_weights(train_loader.dataset, device)
    # Recall kelas cancer lebih kritis secara klinis (false negative lebih berbahaya
    # daripada false positive) -> beri bobot ekstra di atas 'balanced' agar model
    # dihukum lebih keras saat melewatkan kasus cancer.
    CANCER_RECALL_BOOST = 1.3
    cancer_idx_tmp = class_names.index("cancer")
    class_weights[cancer_idx_tmp] *= CANCER_RECALL_BOOST
    print(
        "Class weights (setelah boost recall cancer):",
        {c: round(w, 4) for c, w in zip(class_names, class_weights.tolist())},
        flush=True,
    )

    with open(MODELS_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f)

    # ------------------------------------------------------------------
    # 2. Model + regularisasi
    # ------------------------------------------------------------------
    # v6: training set diperbesar dari 2.012 -> 2.855 dengan menambahkan 80% dari
    # dataset eksternal IQ-OTH/NCCD (data CT asli terverifikasi, bukan duplikat/leakage
    # dari sumber yang sama). Data lebih banyak -> regularisasi bisa dilonggarkan
    # sedikit dari v5 tanpa risiko overfitting yang sama besar.
    DROP_RATE = 0.3
    DROP_PATH_RATE = 0.2
    LABEL_SMOOTHING = 0.1
    WEIGHT_DECAY = 1.2e-4

    model = build_model(
        num_classes=len(class_names),
        pretrained=True,
        drop_rate=DROP_RATE,
        drop_path_rate=DROP_PATH_RATE,
        model_name=MODEL_NAME,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    print(f"Total parameter: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # ------------------------------------------------------------------
    # 3. Training loop dengan monitoring gap overfitting
    # ------------------------------------------------------------------
    def gap_status(train_acc, val_acc):
        gap = train_acc - val_acc
        if gap < 0.05:
            return gap, "OK"
        if gap < 0.15:
            return gap, "WARNING"
        return gap, "OVERFITTING"

    history = {
        "epoch": [], "phase": [], "lr": [],
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "gap": [], "gap_status": [],
    }

    best_val_loss = float("inf")
    BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"

    def run_epochs(phase_name, epochs, optimizer, scheduler, patience):
        nonlocal best_val_loss
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            gap, status = gap_status(train_acc, val_acc)
            current_lr = optimizer.param_groups[0]["lr"]

            history["epoch"].append(len(history["epoch"]) + 1)
            history["phase"].append(phase_name)
            history["lr"].append(current_lr)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["gap"].append(gap)
            history["gap_status"].append(status)

            improved = val_loss < best_val_loss - 1e-4
            if improved:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), BEST_MODEL_PATH)
            else:
                epochs_no_improve += 1

            print(
                f"[{phase_name}] Epoch {epoch}/{epochs} lr={current_lr:.2e} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"gap={gap:+.4f} [{status}] "
                f"{'(saved)' if improved else ''} ({time.time() - t0:.1f}s)",
                flush=True,
            )

            scheduler.step()

            if epochs_no_improve >= patience:
                print(
                    f"Early stopping: val_loss tidak membaik selama {patience} epoch berturut-turut.",
                    flush=True,
                )
                break

    # --- Fase 1: Transfer learning (freeze backbone) ---
    freeze_backbone(model)
    print(f"[Fase1] Parameter trainable (head only): {count_trainable_params(model):,}", flush=True)

    PHASE1_EPOCHS = 8
    PHASE1_PATIENCE = 5

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)
    run_epochs("Fase1-Transfer", PHASE1_EPOCHS, optimizer, scheduler, PHASE1_PATIENCE)

    # --- Fase 2: Fine-tuning (unfreeze semua layer) ---
    unfreeze_all(model)
    print(f"[Fase2] Parameter trainable (semua layer): {count_trainable_params(model):,}", flush=True)

    PHASE2_EPOCHS = 18
    PHASE2_PATIENCE = 6
    WARMUP_EPOCHS = 3

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS - WARMUP_EPOCHS)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
    )
    run_epochs("Fase2-FineTune", PHASE2_EPOCHS, optimizer, scheduler, PHASE2_PATIENCE)

    print(f"Best val_loss selama training: {best_val_loss:.4f}", flush=True)

    # ------------------------------------------------------------------
    # 4. Kurva training
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
    # 5. Evaluasi test set (model terbaik)
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)

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
    # 6. Ringkasan akhir
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
            "class_weights": class_weights.tolist(),
            "phase1_epochs_planned": PHASE1_EPOCHS,
            "phase2_epochs_planned": PHASE2_EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
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
