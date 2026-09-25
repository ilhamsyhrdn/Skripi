"""Out-of-distribution (domain) gate: predicts whether an uploaded image is a
lung CT slice at all, before the cancer classifier runs on it."""
from __future__ import annotations

from pathlib import Path

import torch

from data.ood_dataset import build_ood_transforms
from models.factory import build_model

THRESHOLD = 0.5


class OODDetector:
    def __init__(self, model_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.transform = build_ood_transforms(train=False)
        self.available = Path(model_path).exists()
        if self.available:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model = build_model("efficientnet_b0", n_classes=2).to(self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()

    def predict(self, image):
        if not self.available:
            return {"available": False, "is_lung_ct": True, "prob_lung_ct": 1.0}
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        prob_lung_ct = float(probs[1])
        return {"available": True, "is_lung_ct": prob_lung_ct >= THRESHOLD, "prob_lung_ct": prob_lung_ct}
