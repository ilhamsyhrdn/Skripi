import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))


def main():
    import torch

    from dataset import get_dataloaders
    from model import build_model
    from ood_detector import extract_embeddings, fit_ood_detector, calibrate_threshold, save_ood_detector, ood_scores

    ROOT = SRC_DIR.parent
    DATA_ROOT = ROOT / "dataset_split"
    MODELS_DIR = ROOT / "models"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=2, pretrained=False, model_name="efficientnet_b0").to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pth", map_location=device, weights_only=True))
    model.eval()

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        DATA_ROOT, batch_size=32, num_workers=2, img_size=224
    )

    print("Ekstrak embedding training...")
    train_emb = extract_embeddings(model, train_loader, device)
    print(f"Train embeddings: {train_emb.shape}")

    print("Ekstrak embedding validation...")
    val_emb = extract_embeddings(model, val_loader, device)
    print(f"Val embeddings: {val_emb.shape}")

    print("Ekstrak embedding test (untuk cek false-rejection rate)...")
    test_emb = extract_embeddings(model, test_loader, device)

    K = 5
    detector = fit_ood_detector(train_emb, k=K)
    threshold = calibrate_threshold(detector, val_emb, percentile=99.0)
    print(f"\nThreshold OOD (persentil 99 dari skor val): {threshold:.4f}")

    val_scores = ood_scores(detector, val_emb)
    test_scores = ood_scores(detector, test_emb)
    print(f"Val scores:  min={val_scores.min():.4f} mean={val_scores.mean():.4f} max={val_scores.max():.4f}")
    print(f"Test scores: min={test_scores.min():.4f} mean={test_scores.mean():.4f} max={test_scores.max():.4f}")

    false_rejection_val = float((val_scores > threshold).mean())
    false_rejection_test = float((test_scores > threshold).mean())
    print(f"\nFalse-rejection rate (val, citra CT paru asli ditolak keliru): {false_rejection_val:.4f}")
    print(f"False-rejection rate (test, citra CT paru asli ditolak keliru): {false_rejection_test:.4f}")

    save_ood_detector(detector, threshold, K, MODELS_DIR / "ood_detector.pkl")
    print(f"\nDetektor OOD disimpan ke {MODELS_DIR / 'ood_detector.pkl'}")


if __name__ == "__main__":
    main()
