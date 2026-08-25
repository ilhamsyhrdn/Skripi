"""Pipeline inferensi final: cek OOD (bukan CT paru?) lalu klasifikasi cancer/no_cancer.

Pipeline:
1. Ekstrak fitur penultimate dari citra input (memakai model v6).
2. Hitung skor OOD (jarak k-NN ke embedding training). Kalau di atas threshold,
   tolak sebagai "bukan citra CT paru-paru" -- tidak dipaksakan diklasifikasi.
3. Kalau lolos, klasifikasi lewat ensemble (v5 + v6, rata-rata softmax tanpa TTA
   -- TTA horizontal-flip terbukti menurunkan performa, lihat src/eval_ensemble.py).

Pemakaian CLI:
    python src/predict.py path/to/citra.png
"""

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))


class LungCTPredictor:
    def __init__(self, models_dir: Path = None, device=None):
        import torch

        from dataset import get_transforms
        from model import build_model
        from ood_detector import load_ood_detector

        self.models_dir = models_dir or (SRC_DIR.parent / "models")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = json.loads((self.models_dir / "class_names.json").read_text())
        self.cancer_idx = self.class_names.index("cancer")
        self.transform = get_transforms(train=False, img_size=224)

        def load(path):
            m = build_model(num_classes=len(self.class_names), pretrained=False, model_name="efficientnet_b0")
            m.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            m.to(self.device).eval()
            return m

        self.v5 = load(self.models_dir / "efficientnet_b0_v5_clean_data.pth")
        self.v6 = load(self.models_dir / "efficientnet_b0_v6_80pct.pth")
        self.ood_detector, self.ood_threshold, self.ood_k = load_ood_detector(
            self.models_dir / "ood_detector.pkl"
        )

    def predict(self, image) -> dict:
        """image: path (str/Path) atau PIL.Image sudah dibuka."""
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from ood_detector import ood_scores

        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")

        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.v6.forward_features(x)
            embedding = self.v6.forward_head(features, pre_logits=True).cpu().numpy()
            ood_score = float(ood_scores(self.ood_detector, embedding)[0])

            if ood_score > self.ood_threshold:
                return {
                    "is_valid_ct": False,
                    "ood_score": ood_score,
                    "ood_threshold": self.ood_threshold,
                    "message": "Citra ini terdeteksi BUKAN citra CT paru-paru -- tidak dapat diklasifikasikan.",
                }

            p5 = F.softmax(self.v5(x), dim=1)
            p6 = F.softmax(self.v6(x), dim=1)
            probs = ((p5 + p6) / 2)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        return {
            "is_valid_ct": True,
            "ood_score": ood_score,
            "ood_threshold": self.ood_threshold,
            "predicted_class": self.class_names[pred_idx],
            "probabilities": {c: float(p) for c, p in zip(self.class_names, probs)},
            "cancer_probability": float(probs[self.cancer_idx]),
        }


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <path_to_image>")
        sys.exit(1)

    predictor = LungCTPredictor()
    result = predictor.predict(sys.argv[1])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
