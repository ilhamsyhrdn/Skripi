"""Bandingkan baseline v6, TTA, dan beberapa kombinasi ensemble di test set.

Temuan: TTA (horizontal flip) JUSTRU menurunkan performa (anatomi CT paru
punya orientasi kiri-kanan yang bermakna — jantung, dll — sehingga flip
mengganggu, bukan membantu, meski dipakai sebagai augmentasi saat training).
Ensemble 2 model (v5+v6) tanpa TTA memberi hasil terbaik secara keseluruhan
(accuracy & ROC-AUC naik, recall cancer nyaris tidak turun dari v6 sendiri).
Menambahkan v7 ke ensemble menaikkan accuracy tipis tapi menurunkan recall
cancer cukup banyak, sehingga TIDAK dipakai di kombinasi final.

Kombinasi final yang dipakai di src/predict.py: ensemble (v5 + v6), tanpa TTA.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))


def main():
    import json
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

    from dataset import get_dataloaders
    from model import build_model

    ROOT = SRC_DIR.parent
    DATA_ROOT = ROOT / "dataset_split"
    MODELS_DIR = ROOT / "models"
    REPORTS_DIR = ROOT / "reports"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = json.loads((MODELS_DIR / "class_names.json").read_text())
    cancer_idx = class_names.index("cancer")

    def load_model(path):
        m = build_model(num_classes=len(class_names), pretrained=False, model_name="efficientnet_b0").to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        return m

    v5 = load_model(MODELS_DIR / "efficientnet_b0_v5_clean_data.pth")
    v6 = load_model(MODELS_DIR / "efficientnet_b0_v6_80pct.pth")
    v7 = load_model(MODELS_DIR / "efficientnet_b0_v7_100pct.pth")

    _, _, test_loader, _ = get_dataloaders(DATA_ROOT, batch_size=32, num_workers=2, img_size=224)

    all_labels = []
    probs_v6_only = []
    probs_v6_tta = []
    probs_v5_v6 = []
    probs_v5_v6_v7 = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            flipped = torch.flip(images, dims=[3])

            p5 = F.softmax(v5(images), dim=1)
            p6 = F.softmax(v6(images), dim=1)
            p7 = F.softmax(v7(images), dim=1)
            p6_flip = F.softmax(v6(flipped), dim=1)

            probs_v6_only.append(p6.cpu().numpy())
            probs_v6_tta.append(((p6 + p6_flip) / 2).cpu().numpy())
            probs_v5_v6.append(((p5 + p6) / 2).cpu().numpy())
            probs_v5_v6_v7.append(((p5 + p6 + p7) / 3).cpu().numpy())
            all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels)

    def evaluate(probs, name):
        probs = np.concatenate(probs, axis=0)
        preds = probs.argmax(axis=1)
        acc = float((preds == all_labels).mean())
        binary_labels = (all_labels == cancer_idx).astype(int)
        auc = float(roc_auc_score(binary_labels, probs[:, cancer_idx]))
        ap = float(average_precision_score(binary_labels, probs[:, cancer_idx]))
        report = classification_report(all_labels, preds, target_names=class_names, digits=4)
        cm = confusion_matrix(all_labels, preds)
        print(f"\n===== {name} =====")
        print(f"Accuracy: {acc:.4f}  ROC-AUC: {auc:.4f}  PR-AUC: {ap:.4f}")
        print(report)
        print("Confusion matrix:\n", cm)
        return {"accuracy": acc, "roc_auc": auc, "pr_auc": ap}

    results = {
        "baseline_v6_only": evaluate(probs_v6_only, "Baseline: v6 saja"),
        "v6_plus_tta": evaluate(probs_v6_tta, "v6 + TTA flip (tanpa ensemble) -- lebih buruk, dibuang"),
        "ensemble_v5_v6": evaluate(probs_v5_v6, "Ensemble v5+v6, tanpa TTA -- FINAL"),
        "ensemble_v5_v6_v7": evaluate(probs_v5_v6_v7, "Ensemble v5+v6+v7, tanpa TTA -- recall turun, tidak dipakai"),
    }

    with open(REPORTS_DIR / "ensemble_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDisimpan ke reports/ensemble_comparison.json")


if __name__ == "__main__":
    main()
