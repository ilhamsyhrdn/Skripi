"""Two-phase transfer learning (feature extraction -> fine-tuning) for
EfficientNet-B0 and ResNet50, 5-fold StratifiedGroupKFold. Trains
5 folds x 2 architectures = 10 models per --variant, each saved as
outputs/models_<variant>/{arch}_fold{k}.pt (keeping the best val
macro-F1 checkpoint seen across BOTH phases).

--variant selects which experiment stage this run corresponds to
(Bab IV): whole_slice (LIDC-IDRI, no crop) -> crop (LIDC-IDRI, nodule
crop) -> pathology (LIDC-IDRI, pathology-corrected labels) -> combined
(Kaggle + LIDC-IDRI, the adopted final configuration).

Live per-epoch progress is printed to stdout (flush=True) so the run can be
watched in real time, not silently.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import CLASS_NAMES, LungCTDataset, build_transforms
from models.factory import build_model, freeze_backbone, unfreeze_for_finetune

MANIFESTS = Path("D:/skripsi/Kodingan/outputs/manifests")
OUTPUTS = Path("D:/skripsi/Kodingan/outputs")
REPORT_DIR = OUTPUTS / "reports"

# variant -> (trainval csv, model dir, history filename prefix)
VARIANTS = {
    "whole_slice": (MANIFESTS / "trainval_folds.csv", OUTPUTS / "models", ""),
    "crop": (MANIFESTS / "trainval_folds_crop.csv", OUTPUTS / "models_crop", "crop_"),
    "pathology": (MANIFESTS / "trainval_folds_final.csv", OUTPUTS / "models_final", "final_"),
    "combined": (MANIFESTS / "trainval_folds_combined.csv", OUTPUTS / "models_combined", "combined_"),
    # uncropped chest slices: a nodule occupies few pixels, so the input
    # resolution is raised to keep that detail after resizing
    "full": (MANIFESTS / "trainval_folds_full.csv", OUTPUTS / "models_full", "full_"),
}

N_FOLDS = 5
BATCH_SIZE = 32
IMG_SIZE = 224


def class_weights_from_df(df, device):
    counts = df["canonical_label"].value_counts()
    freqs = np.array([counts.get(c, 1) for c in CLASS_NAMES], dtype=np.float64)
    weights = freqs.sum() / (len(CLASS_NAMES) * freqs)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_losses = [], [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            all_losses.append(loss.item())
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    malignant_idx = CLASS_NAMES.index("Malignant")
    cancer_recall = recall_score(all_labels, all_preds, labels=[malignant_idx], average="macro", zero_division=0)
    return {"loss": float(np.mean(all_losses)), "acc": acc, "macro_f1": macro_f1, "cancer_recall": cancer_recall}


def fit(model, train_loader, val_loader, criterion, optimizer, device, epochs, phase_name,
        scaler, patience=6, scheduler=None, best_state_holder=None, history=None):
    best_f1_this_phase = -1.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_losses, train_correct, train_total = [], 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)

        train_loss = float(np.mean(train_losses))
        train_acc = train_correct / train_total
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        if scheduler is not None:
            scheduler.step(val_metrics["macro_f1"])
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"  [{phase_name}] epoch {epoch}/{epochs} lr={lr_now:.1e} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} "
              f"val_macroF1={val_metrics['macro_f1']:.3f} val_cancerRecall={val_metrics['cancer_recall']:.3f} "
              f"({elapsed:.1f}s)", flush=True)

        if history is not None:
            history.append({"phase": phase_name, "epoch": epoch, "lr": lr_now,
                             "train_loss": train_loss, "train_acc": train_acc, **val_metrics})

        if best_state_holder is not None and val_metrics["macro_f1"] > best_state_holder["best_f1"]:
            best_state_holder["best_f1"] = val_metrics["macro_f1"]
            best_state_holder["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if val_metrics["macro_f1"] > best_f1_this_phase:
            best_f1_this_phase = val_metrics["macro_f1"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  [{phase_name}] early stopping at epoch {epoch} "
                      f"(no val macro-F1 improvement for {patience} epochs)", flush=True)
                break


def train_one_fold(arch: str, fold: int, trainval_df: pd.DataFrame, device, model_dir: Path,
                   history_prefix: str, img_size: int = IMG_SIZE, batch_size: int = BATCH_SIZE,
                   augment: bool = True, augment_strength: str = "medium"):
    train_df = trainval_df[trainval_df["fold"] != fold].reset_index(drop=True)
    val_df = trainval_df[trainval_df["fold"] == fold].reset_index(drop=True)

    # augment=False gives the deterministic pipeline (resize + normalise only),
    # which is the ablation used to measure what augmentation actually buys
    train_ds = LungCTDataset(train_df, build_transforms(img_size, train=augment,
                                                        augment_strength=augment_strength))
    val_ds = LungCTDataset(val_df, build_transforms(img_size, train=False))
    # data loading, not GPU compute, is the bottleneck here -- more workers help
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=6,
                              pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True)

    model = build_model(arch).to(device)
    class_weights = class_weights_from_df(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    best_holder = {"best_f1": -1.0, "state_dict": None}
    history = []

    print(f"\n=== {arch} fold {fold} === train={len(train_df)} val={len(val_df)} "
          f"class_weights={class_weights.cpu().numpy().round(3).tolist()}", flush=True)

    # Phase A: feature extraction
    freeze_backbone(model, arch)
    opt_a = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    fit(model, train_loader, val_loader, criterion, opt_a, device,
        epochs=12, phase_name="A-head", scaler=scaler, patience=6,
        best_state_holder=best_holder, history=history)

    # Phase B: fine-tuning
    unfreeze_for_finetune(model, arch, n_blocks=3)
    opt_b = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    sched_b = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_b, mode="max", factor=0.5, patience=3)
    fit(model, train_loader, val_loader, criterion, opt_b, device,
        epochs=25, phase_name="B-finetune", scaler=scaler, scheduler=sched_b,
        patience=6, best_state_holder=best_holder, history=history)

    model.load_state_dict(best_holder["state_dict"])
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": arch, "fold": fold,
                "best_val_macro_f1": best_holder["best_f1"]},
               model_dir / f"{arch}_fold{fold}.pt")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_json(REPORT_DIR / f"history_{history_prefix}{arch}_fold{fold}.json", orient="records")
    print(f"=== {arch} fold {fold} DONE -- best val macro-F1 = {best_holder['best_f1']:.4f} ===", flush=True)
    return best_holder["best_f1"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="combined",
                         help="Which experiment stage (Bab IV) to train.")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE,
                         help="Input resolution; raise it for uncropped slices.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-augment", action="store_true",
                         help="Ablation: train without data augmentation, to measure its effect.")
    parser.add_argument("--augment-strength", default="medium", choices=["ct", "light", "medium", "heavy"],
                         help="Which augmentation preset to train with (see data/dataset.py).")
    args = parser.parse_args()
    trainval_csv, model_dir, history_prefix = VARIANTS[args.variant]
    if args.no_augment:
        model_dir = model_dir.with_name(model_dir.name + "_noaug")
        history_prefix = history_prefix + "noaug_"
    elif args.augment_strength != "medium":
        model_dir = model_dir.with_name(f"{model_dir.name}_aug{args.augment_strength}")
        history_prefix = f"{history_prefix}aug{args.augment_strength}_"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Variant: {args.variant} | Device: {device} | img_size={args.img_size} | "
          f"batch={args.batch_size} | "
          f"augmentasi={'TIDAK' if args.no_augment else args.augment_strength}", flush=True)
    trainval_df = pd.read_csv(trainval_csv)

    results = {}
    for arch in ["efficientnet_b0", "resnet50"]:
        for fold in range(N_FOLDS):
            # a finished checkpoint is left alone, so an interrupted run resumes
            # where it stopped instead of retraining everything from scratch
            ckpt = model_dir / f"{arch}_fold{fold}.pt"
            if ckpt.exists():
                prev = torch.load(ckpt, map_location="cpu", weights_only=False)
                results[f"{arch}_fold{fold}"] = prev.get("best_val_macro_f1", float("nan"))
                print(f"\n=== {arch} fold {fold} DILEWATI (checkpoint sudah ada) ===", flush=True)
                continue
            f1 = train_one_fold(arch, fold, trainval_df, device, model_dir, history_prefix,
                                img_size=args.img_size, batch_size=args.batch_size,
                                augment=not args.no_augment,
                                augment_strength=args.augment_strength)
            results[f"{arch}_fold{fold}"] = f1

    print("\n=== ALL FOLDS DONE ===", flush=True)
    for k, v in results.items():
        print(f"  {k}: best val macro-F1 = {v:.4f}", flush=True)


if __name__ == "__main__":
    main()
