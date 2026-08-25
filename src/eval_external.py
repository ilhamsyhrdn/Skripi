import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))


def main():
    import numpy as np
    import torch
    from PIL import Image
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        average_precision_score,
        precision_recall_curve,
    )

    from dataset import get_transforms
    from model import build_model

    ROOT = SRC_DIR.parent
    EXTERNAL_ROOT = (
        ROOT
        / "dataset_external"
        / "iqothnccd"
        / "The IQ-OTHNCCD lung cancer dataset"
        / "The IQ-OTHNCCD lung cancer dataset"
    )
    REPORTS_DIR = ROOT / "reports"
    MODELS_DIR = ROOT / "models"

    class_names = json.loads((MODELS_DIR / "class_names.json").read_text())
    cancer_idx = class_names.index("cancer")
    no_cancer_idx = class_names.index("no_cancer")

    # Malignant = cancer secara klinis. Benign (bukan kanker) + Normal (tidak ada nodul)
    # dipetakan ke no_cancer.
    folder_to_label = {
        "Malignant cases": cancer_idx,
        "Bengin cases": no_cancer_idx,
        "Normal cases": no_cancer_idx,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names), pretrained=False, model_name="efficientnet_b0").to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pth", map_location=device, weights_only=True))
    model.eval()

    transform = get_transforms(train=False, img_size=224)

    records = []
    for folder_name, label in folder_to_label.items():
        for path in sorted((EXTERNAL_ROOT / folder_name).glob("*.jpg")):
            records.append((path, label, folder_name))

    print(f"Total citra eksternal (IQ-OTH/NCCD): {len(records)}")

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for path, label, _ in records:
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(probs.argmax().item())
            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(float(probs[cancer_idx].item()))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = float((all_preds == all_labels).mean())
    print(f"\nExternal accuracy: {acc:.4f}")

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    binary_labels = (all_labels == cancer_idx).astype(int)
    auc = roc_auc_score(binary_labels, all_probs)
    ap = average_precision_score(binary_labels, all_probs)
    print(f"ROC-AUC: {auc:.4f}  PR-AUC: {ap:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:\n", cm)

    per_folder = {}
    for folder_name in folder_to_label:
        idxs = [i for i, r in enumerate(records) if r[2] == folder_name]
        folder_preds = all_preds[idxs]
        folder_labels = all_labels[idxs]
        per_folder[folder_name] = {
            "n": len(idxs),
            "acc": float((folder_preds == folder_labels).mean()),
            "pred_cancer_rate": float((folder_preds == cancer_idx).mean()),
        }
    print("\nPer-folder breakdown:")
    print(json.dumps(per_folder, indent=2))

    precisions, recalls, thresholds = precision_recall_curve(binary_labels, all_probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_i = int(np.argmax(f1s[:-1]))
    recalibrated_threshold = float(thresholds[best_i])
    recalibrated_preds = (all_probs >= recalibrated_threshold).astype(int)
    recalibrated_acc = float((recalibrated_preds == binary_labels).mean())
    print(
        f"\nSetelah kalibrasi ulang threshold (domain shift): thr={recalibrated_threshold:.4f} "
        f"precision={precisions[best_i]:.4f} recall={recalls[best_i]:.4f} f1={f1s[best_i]:.4f} "
        f"accuracy={recalibrated_acc:.4f}"
    )

    summary = {
        "dataset": "IQ-OTH/NCCD (external, CT, verified)",
        "n_images": len(records),
        "label_mapping": {"Malignant cases": "cancer", "Bengin cases": "no_cancer", "Normal cases": "no_cancer"},
        "external_accuracy": acc,
        "external_roc_auc": auc,
        "external_pr_auc": ap,
        "confusion_matrix": cm.tolist(),
        "per_folder": per_folder,
        "recalibrated_threshold": recalibrated_threshold,
        "recalibrated_accuracy": recalibrated_acc,
        "recalibrated_precision": float(precisions[best_i]),
        "recalibrated_recall": float(recalls[best_i]),
        "recalibrated_f1": float(f1s[best_i]),
    }
    with open(REPORTS_DIR / "external_validation_iqothnccd.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nDisimpan ke reports/external_validation_iqothnccd.json")


if __name__ == "__main__":
    main()
