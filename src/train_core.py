"""Logika training inti (2 fase: transfer learning + fine-tuning) yang dipakai
bersama oleh src/train.py (single split) dan src/kfold_train.py (5-fold CV).
"""

import time


def run_training(
    train_loader,
    val_loader,
    class_names,
    device,
    best_model_path,
    model_name="efficientnet_b0",
    drop_rate=0.3,
    drop_path_rate=0.2,
    label_smoothing=0.1,
    weight_decay=1.2e-4,
    cancer_recall_boost=1.3,
    boost_class_name="cancer",
    phase1_epochs=8,
    phase1_patience=5,
    phase2_epochs=18,
    phase2_patience=6,
    warmup_epochs=3,
    log_prefix="",
    model=None,
):
    """model: kalau diberikan, dipakai langsung (backbone selain EfficientNet-B0
    lewat timm, misal RadImageNetClassifier) -- model_name/drop_rate/drop_path_rate
    diabaikan dalam kasus ini. Kalau None, dibangun lewat build_model() seperti biasa.

    boost_class_name: nama kelas yang recall-nya mau ditekankan lewat class_weight
    ekstra (cancer_recall_boost). Kalau nama kelasnya tidak ada di class_names atau
    cancer_recall_boost==1.0, boost dilewati (tidak error).
    """
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    from dataset import compute_class_weights
    from model import build_model, freeze_backbone, unfreeze_all, count_trainable_params
    from engine import train_one_epoch, evaluate

    class_weights = compute_class_weights(train_loader.dataset, device)
    if cancer_recall_boost != 1.0 and boost_class_name in class_names:
        class_weights[class_names.index(boost_class_name)] *= cancer_recall_boost

    if model is None:
        model = build_model(
            num_classes=len(class_names),
            pretrained=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            model_name=model_name,
        )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

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
    state = {"best_val_loss": float("inf")}

    def run_epochs(phase_name, epochs, optimizer, scheduler, patience):
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

            improved = val_loss < state["best_val_loss"] - 1e-4
            if improved:
                state["best_val_loss"] = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1

            print(
                f"{log_prefix}[{phase_name}] Epoch {epoch}/{epochs} lr={current_lr:.2e} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"gap={gap:+.4f} [{status}] "
                f"{'(saved)' if improved else ''} ({time.time() - t0:.1f}s)",
                flush=True,
            )

            scheduler.step()
            if epochs_no_improve >= patience:
                print(
                    f"{log_prefix}Early stopping: val_loss tidak membaik selama {patience} epoch berturut-turut.",
                    flush=True,
                )
                break

    freeze_backbone(model)
    print(f"{log_prefix}[Fase1] Parameter trainable (head only): {count_trainable_params(model):,}", flush=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs)
    run_epochs("Fase1-Transfer", phase1_epochs, optimizer, scheduler, phase1_patience)

    unfreeze_all(model)
    print(f"{log_prefix}[Fase2] Parameter trainable (semua layer): {count_trainable_params(model):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=weight_decay)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs - warmup_epochs)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    run_epochs("Fase2-FineTune", phase2_epochs, optimizer, scheduler, phase2_patience)

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.to(device)

    return {
        "model": model,
        "history": history,
        "best_val_loss": state["best_val_loss"],
        "class_weights": class_weights.tolist(),
    }
