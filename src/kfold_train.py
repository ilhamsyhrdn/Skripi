"""5-fold cross-validation + ensemble.

Menggabungkan seluruh train+val pool (test TIDAK disentuh sama sekali) jadi satu
kumpulan, dibagi 5 fold stratified. Tiap fold dilatih dengan konfigurasi yang sama
persis dengan v6 (lihat src/train_core.py), lalu kelima model di-ensemble (rata-rata
softmax) dan dievaluasi sekali di test set yang sudah ada.

Alasan teknik ini dipilih: memakai seluruh data train+val lebih menyeluruh (setiap
citra pernah jadi bagian training di 4 dari 5 fold), dan ensemble 5 model yang
masing-masing melihat subset data sedikit berbeda cenderung menurunkan variansi
prediksi dibanding 1 model tunggal -- tanpa menambah risiko overfitting karena tidak
ada model yang lebih besar/kompleks, hanya lebih banyak model yang lebih kecil
digabung.
"""

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

N_FOLDS = 5
RANDOM_SEED = 42


class ManifestImageDataset:
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image

        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def build_manifest(data_root, class_names):
    records = []
    for split in ["train", "val"]:
        for label_idx, label in enumerate(class_names):
            folder = data_root / split / label
            for p in sorted(folder.glob("*")):
                if p.is_file():
                    records.append((str(p), label_idx))
    return records


def main():
    import random

    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

    from dataset import get_transforms, get_dataloaders
    from train_core import run_training

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

    class_names = json.loads((MODELS_DIR / "class_names.json").read_text())
    cancer_idx = class_names.index("cancer")

    records = build_manifest(DATA_ROOT, class_names)
    paths = [r[0] for r in records]
    labels = [r[1] for r in records]
    print(f"Total pool (train+val digabung) untuk k-fold: {len(paths)}", flush=True)

    _, _, test_loader, _ = get_dataloaders(DATA_ROOT, batch_size=32, num_workers=2, img_size=224)
    print(f"Test set (tidak disentuh k-fold): {len(test_loader.dataset)}", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_summaries = []
    fold_model_paths = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(paths, labels), start=1):
        print(f"\n{'='*20} FOLD {fold_idx}/{N_FOLDS} {'='*20}", flush=True)

        train_paths = [paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_paths = [paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        print(f"Fold train: {len(train_paths)}  Fold val: {len(val_paths)}", flush=True)

        train_ds = ManifestImageDataset(train_paths, train_labels, get_transforms(True, 224))
        val_ds = ManifestImageDataset(val_paths, val_labels, get_transforms(False, 224))
        train_ds.targets = train_labels  # dibutuhkan oleh compute_class_weights

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

        fold_model_path = MODELS_DIR / f"kfold_fold{fold_idx}.pth"
        result = run_training(
            train_loader, val_loader, class_names, device, fold_model_path,
            log_prefix=f"[Fold {fold_idx}] ",
        )
        fold_model_paths.append(fold_model_path)
        fold_summaries.append({"fold": fold_idx, "best_val_loss": result["best_val_loss"], "n_train": len(train_paths), "n_val": len(val_paths)})
        print(f"[Fold {fold_idx}] Selesai. Best val_loss: {result['best_val_loss']:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Evaluasi ensemble 5-fold di test set
    # ------------------------------------------------------------------
    print(f"\n{'='*20} EVALUASI ENSEMBLE 5-FOLD DI TEST SET {'='*20}", flush=True)

    from model import build_model

    fold_models = []
    for p in fold_model_paths:
        m = build_model(num_classes=len(class_names), pretrained=False, model_name="efficientnet_b0").to(device)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.eval()
        fold_models.append(m)

    all_labels = []
    ensemble_probs = []
    per_fold_probs = [[] for _ in fold_models]

    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(device)
            probs_sum = torch.zeros(images.size(0), len(class_names), device=device)
            for i, m in enumerate(fold_models):
                p = F.softmax(m(images), dim=1)
                probs_sum += p
                per_fold_probs[i].append(p.cpu().numpy())
            ensemble_probs.append((probs_sum / len(fold_models)).cpu().numpy())
            all_labels.extend(labels_batch.numpy().tolist())

    all_labels = np.array(all_labels)
    ensemble_probs = np.concatenate(ensemble_probs, axis=0)
    binary_labels = (all_labels == cancer_idx).astype(int)

    ensemble_preds = ensemble_probs.argmax(axis=1)
    ensemble_acc = float((ensemble_preds == all_labels).mean())
    ensemble_auc = float(roc_auc_score(binary_labels, ensemble_probs[:, cancer_idx]))
    ensemble_ap = float(average_precision_score(binary_labels, ensemble_probs[:, cancer_idx]))
    report = classification_report(all_labels, ensemble_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, ensemble_preds)

    print(f"\nEnsemble 5-fold -> Accuracy: {ensemble_acc:.4f}  ROC-AUC: {ensemble_auc:.4f}  PR-AUC: {ensemble_ap:.4f}", flush=True)
    print(report, flush=True)
    print("Confusion matrix:\n", cm, flush=True)

    per_fold_accs = []
    for i, probs_list in enumerate(per_fold_probs, start=1):
        probs = np.concatenate(probs_list, axis=0)
        preds = probs.argmax(axis=1)
        acc = float((preds == all_labels).mean())
        per_fold_accs.append(acc)
        print(f"Fold {i} solo test accuracy: {acc:.4f}", flush=True)

    summary = {
        "n_folds": N_FOLDS,
        "fold_summaries": fold_summaries,
        "per_fold_test_accuracy": per_fold_accs,
        "ensemble_test_accuracy": ensemble_acc,
        "ensemble_roc_auc": ensemble_auc,
        "ensemble_pr_auc": ensemble_ap,
        "ensemble_confusion_matrix": cm.tolist(),
    }
    with open(REPORTS_DIR / "kfold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(REPORTS_DIR / "kfold_classification_report.txt", "w") as f:
        f.write(report)

    print("\nDisimpan ke reports/kfold_summary.json", flush=True)


if __name__ == "__main__":
    main()
