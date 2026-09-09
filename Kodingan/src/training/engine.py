"""Train/eval loops + metrics. Prints live per-epoch progress (train loss/acc,
val loss/acc/macro-F1/cancer-recall) -- never a silent long-running fit() call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.data.dataset import CLASS_NAMES


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    macro_f1: float
    malignant_recall: float
    per_class_recall: dict = field(default_factory=dict)
    confusion: np.ndarray | None = None
    probs: np.ndarray | None = None
    labels: np.ndarray | None = None


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n_seen = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                logits = model(x)
                loss = criterion(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                logits = model(x)
                loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        n_seen += x.size(0)
        probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(probs.argmax(axis=1))
        all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    acc = float((all_preds == all_labels).mean())
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_recall_arr = recall_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(len(CLASS_NAMES))))
    per_class_recall = dict(zip(CLASS_NAMES, per_class_recall_arr.tolist()))
    malignant_idx = CLASS_NAMES.index("Malignant")
    malignant_recall = float(per_class_recall_arr[malignant_idx])
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))

    return EpochResult(
        loss=total_loss / max(n_seen, 1),
        accuracy=acc,
        macro_f1=float(macro_f1),
        malignant_recall=malignant_recall,
        per_class_recall=per_class_recall,
        confusion=cm,
        probs=all_probs,
        labels=all_labels,
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    phase_name: str,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler=None,
    patience: int = 7,
    best_state_holder: dict | None = None,
    log_fn=print,
) -> list[dict]:
    """Runs `epochs` epochs, prints a live line per epoch, early-stops on val macro-F1,
    and (if best_state_holder is given) keeps the best-so-far state_dict in-place so the
    caller can restore it after fine-tuning both phases."""
    history = []
    best_f1 = best_state_holder.get("best_f1", -1.0) if best_state_holder is not None else -1.0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_res = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_res = run_epoch(model, val_loader, criterion, device, None, scaler)
        if scheduler is not None:
            scheduler.step(val_res.macro_f1)
        dt = time.time() - t0

        row = {
            "phase": phase_name,
            "epoch": epoch,
            "train_loss": train_res.loss,
            "train_acc": train_res.accuracy,
            "val_loss": val_res.loss,
            "val_acc": val_res.accuracy,
            "val_macro_f1": val_res.macro_f1,
            "val_malignant_recall": val_res.malignant_recall,
            "val_per_class_recall": val_res.per_class_recall,
            "seconds": round(dt, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        log_fn(
            f"    [{phase_name}] epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_res.loss:.4f} acc={train_res.accuracy:.3f} | "
            f"val_loss={val_res.loss:.4f} acc={val_res.accuracy:.3f} "
            f"macroF1={val_res.macro_f1:.3f} malignantRecall={val_res.malignant_recall:.3f} | "
            f"lr={row['lr']:.2e} | {dt:.1f}s",
            flush=True,
        )

        if val_res.macro_f1 > best_f1:
            best_f1 = val_res.macro_f1
            bad_epochs = 0
            if best_state_holder is not None:
                best_state_holder["state_dict"] = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
                best_state_holder["best_f1"] = best_f1
                best_state_holder["best_epoch"] = (phase_name, epoch)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log_fn(f"    [{phase_name}] early stopping at epoch {epoch} (no val_macro_f1 improvement for {patience} epochs)", flush=True)
                break

    return history
