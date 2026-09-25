"""Train the OOD detector: EfficientNet-B0 feature-extraction-only (backbone
frozen), binary head (Lung CT vs Bukan Lung CT)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.ood_dataset import OODDataset, build_ood_transforms
from models.factory import build_model, freeze_backbone

MANIFEST = Path("D:/skripsi/Kodingan/outputs/manifests/ood_manifest.csv")
MODEL_OUT = Path("D:/skripsi/Kodingan/outputs/models/ood_detector.pt")
REPORT_OUT = Path("D:/skripsi/Kodingan/outputs/reports/ood_training_history.json")
BATCH_SIZE = 32
MAX_EPOCHS = 12


def evaluate(model, loader, device):
    model.eval()
    correct, total, losses = 0, 0, []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(criterion(logits, y).item())
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return {"loss": float(np.mean(losses)), "acc": correct / total}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(MANIFEST)
    train_df, val_df = df[df.split == "train"], df[df.split == "val"]
    print(f"train={len(train_df)} val={len(val_df)} device={device}", flush=True)

    train_ds = OODDataset(train_df, build_ood_transforms(train=True))
    val_ds = OODDataset(val_df, build_ood_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model("efficientnet_b0", n_classes=2).to(device)
    freeze_backbone(model, "efficientnet_b0")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state, history = -1.0, None, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch}/{MAX_EPOCHS} val_acc={val_metrics['acc']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} ({time.time()-t0:.1f}s)", flush=True)
        history.append({"epoch": epoch, **val_metrics})
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if best_acc >= 0.999:
            print("val acc saturated, stopping early", flush=True)
            break

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "best_val_acc": best_acc}, MODEL_OUT)
    pd.DataFrame(history).to_json(REPORT_OUT, orient="records")
    print(f"Saved: {MODEL_OUT} (best_val_acc={best_acc:.4f})", flush=True)


if __name__ == "__main__":
    main()
