"""
5-fold cross-validation training of a transfer-learned, fine-tuned classifier
(EfficientNet-B0 primary / ResNet50 for ensemble diversity) on the deduplicated,
leakage-safe lung-CT manifest.

Two-phase schedule per fold (proposal section 2.4 / 3.4.1 / 3.5.2):
  Phase A - freeze the ImageNet backbone, train only the new 3-class head.
  Phase B - unfreeze the last few backbone blocks, fine-tune everything with a much
            smaller learning rate.
Both phases use early stopping + checkpointing on validation macro-F1.

Prints live per-epoch progress to stdout AND appends every line to
Kodingan/outputs/logs/train_<arch>.log so progress can be tailed while it runs.

Run:
  Kodingan/.venv/Scripts/python.exe -m src.training.train_cv --arch efficientnet_b0
  Kodingan/.venv/Scripts/python.exe -m src.training.train_cv --arch resnet50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

KODINGAN_DIR = Path(__file__).resolve().parents[2]  # .../Kodingan
sys.path.insert(0, str(KODINGAN_DIR))

from src.data.dataset import CLASS_NAMES, LungCTDataset, build_transforms, load_manifest  # noqa: E402
from src.models.factory import build_model, freeze_backbone, trainable_param_count, unfreeze_for_finetune  # noqa: E402
from src.training.engine import fit, run_epoch  # noqa: E402

MANIFEST_DIR = KODINGAN_DIR / "outputs" / "manifests"
MODEL_DIR = KODINGAN_DIR / "outputs" / "models"
LOG_DIR = KODINGAN_DIR / "outputs" / "logs"
REPORT_DIR = KODINGAN_DIR / "outputs" / "reports"
for d in (MODEL_DIR, LOG_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def make_logger(arch: str):
    log_path = LOG_DIR / f"train_{arch}.log"
    fh = open(log_path, "a", encoding="utf-8")

    def log(msg: str = "", flush: bool = True):
        print(msg, flush=flush)
        fh.write(str(msg) + "\n")
        if flush:
            fh.flush()

    return log, fh


def class_weights_from_df(df, device) -> torch.Tensor:
    counts = df["canonical_label"].value_counts()
    freqs = np.array([counts.get(c, 1) for c in CLASS_NAMES], dtype=np.float64)
    weights = freqs.sum() / (len(CLASS_NAMES) * freqs)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--phase_a_epochs", type=int, default=12)
    ap.add_argument("--phase_b_epochs", type=int, default=25)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_finetune", type=float, default=1e-5)
    ap.add_argument("--unfreeze_blocks", type=int, default=3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--augment", default="medium", choices=["light", "medium", "heavy"])
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log, fh = make_logger(args.arch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"\n{'='*90}")
    log(f"START run arch={args.arch} device={device} args={vars(args)}")
    log(f"{'='*90}")

    trainval = load_manifest(MANIFEST_DIR / "trainval_folds.csv")
    test_df = load_manifest(MANIFEST_DIR / "test_holdout.csv")

    folds = [int(f) for f in args.folds.split(",")]
    all_fold_summaries = []

    for k in folds:
        t_fold0 = time.time()
        log(f"\n--- FOLD {k} ---")
        train_df = trainval[trainval["fold"] != k].reset_index(drop=True)
        val_df = trainval[trainval["fold"] == k].reset_index(drop=True)
        log(f"  train={len(train_df)}  val={len(val_df)}  "
            f"train_label_counts={train_df['canonical_label'].value_counts().to_dict()}")

        train_tf = build_transforms(args.image_size, train=True, augment_strength=args.augment)
        eval_tf = build_transforms(args.image_size, train=False)

        train_ds = LungCTDataset(train_df, train_tf)
        val_ds = LungCTDataset(val_df, eval_tf)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        model = build_model(args.arch, dropout=args.dropout).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_from_df(train_df, device))
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        best_holder = {"best_f1": -1.0}

        # ---- Phase A: frozen backbone, train head only ----
        freeze_backbone(model, args.arch)
        trainable, total = trainable_param_count(model)
        log(f"  Phase A (head only): trainable={trainable:,}/{total:,} params")
        opt_a = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr_head)
        hist_a = fit(
            model, train_loader, val_loader, criterion, opt_a, device,
            epochs=args.phase_a_epochs, phase_name="A-head", scaler=scaler,
            patience=args.patience, best_state_holder=best_holder, log_fn=log,
        )

        # restore best-of-phase-A before unfreezing, so phase B fine-tunes from the best point
        model.load_state_dict(best_holder["state_dict"])

        # ---- Phase B: unfreeze last blocks, fine-tune with small LR ----
        unfreeze_for_finetune(model, args.arch, n_blocks=args.unfreeze_blocks)
        trainable, total = trainable_param_count(model)
        log(f"  Phase B (fine-tune last {args.unfreeze_blocks} blocks): trainable={trainable:,}/{total:,} params")
        opt_b = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr_finetune)
        sched_b = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_b, mode="max", factor=0.5, patience=3)
        hist_b = fit(
            model, train_loader, val_loader, criterion, opt_b, device,
            epochs=args.phase_b_epochs, phase_name="B-finetune", scaler=scaler, scheduler=sched_b,
            patience=args.patience, best_state_holder=best_holder, log_fn=log,
        )

        # restore overall best (across both phases) and save
        model.load_state_dict(best_holder["state_dict"])
        ckpt_path = MODEL_DIR / f"{args.arch}_fold{k}.pt"
        torch.save({"state_dict": model.state_dict(), "arch": args.arch, "fold": k,
                    "best_val_macro_f1": best_holder["best_f1"], "best_epoch": best_holder["best_epoch"],
                    "class_names": CLASS_NAMES}, ckpt_path)
        log(f"  saved best checkpoint -> {ckpt_path} (val_macro_f1={best_holder['best_f1']:.4f}, "
            f"from {best_holder['best_epoch']})")

        # quick held-out-test readout for progress visibility (final ensemble number computed separately)
        test_tf = build_transforms(args.image_size, train=False)
        test_ds = LungCTDataset(test_df, test_tf)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_res = run_epoch(model, test_loader, criterion, device, None, scaler)
        log(f"  [fold {k}] held-out TEST (single-model, informational only): "
            f"acc={test_res.accuracy:.3f} macroF1={test_res.macro_f1:.3f} "
            f"malignantRecall={test_res.malignant_recall:.3f} per_class_recall={test_res.per_class_recall}")

        with open(REPORT_DIR / f"{args.arch}_fold{k}_history.json", "w", encoding="utf-8") as jf:
            json.dump(
                {
                    "arch": args.arch, "fold": k,
                    "history": hist_a + hist_b,
                    "best_val_macro_f1": best_holder["best_f1"],
                    "best_epoch": list(best_holder["best_epoch"]),
                    "test_readout": {
                        "accuracy": test_res.accuracy, "macro_f1": test_res.macro_f1,
                        "malignant_recall": test_res.malignant_recall,
                        "per_class_recall": test_res.per_class_recall,
                        "confusion_matrix": test_res.confusion.tolist(),
                    },
                },
                jf, indent=2,
            )

        all_fold_summaries.append({
            "fold": k, "val_macro_f1": best_holder["best_f1"],
            "test_acc": test_res.accuracy, "test_macro_f1": test_res.macro_f1,
            "test_malignant_recall": test_res.malignant_recall,
            "seconds": round(time.time() - t_fold0, 1),
        })
        log(f"  fold {k} done in {time.time()-t_fold0:.1f}s")

        del model
        torch.cuda.empty_cache()

    log(f"\n=== ALL FOLDS DONE ({args.arch}) ===")
    for s in all_fold_summaries:
        log(f"  {s}")
    with open(REPORT_DIR / f"{args.arch}_cv_summary.json", "w", encoding="utf-8") as jf:
        json.dump(all_fold_summaries, jf, indent=2)
    fh.close()


if __name__ == "__main__":
    main()
