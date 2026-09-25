"""Ensemble predictor: loads the 10 trained checkpoints (5xEfficientNet-B0 +
5xResNet50) and combines their softmax probabilities via soft-voting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.dataset import CLASS_NAMES, build_transforms
from models.factory import build_model

N_FOLDS = 5


class EnsemblePredictor:
    def __init__(self, model_dir, device=None, archs=None, img_size=224):
        # img_size must match the resolution the checkpoints were trained at:
        # feeding 224px images to a model fine-tuned at 512px silently wrecks
        # accuracy, so callers pass it explicitly per variant.
        self.archs = archs or ["efficientnet_b0", "resnet50"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.img_size = img_size
        self.transform = build_transforms(img_size, train=False)
        self.models, self.val_f1 = {}, {}
        for arch in self.archs:
            for k in range(N_FOLDS):
                ckpt_path = Path(model_dir) / f"{arch}_fold{k}.pt"
                if not ckpt_path.exists():
                    continue
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model = build_model(arch).to(self.device)
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                key = f"{arch}_fold{k}"
                self.models[key] = model
                self.val_f1[key] = ckpt.get("best_val_macro_f1", 1.0)
        self.member_keys = list(self.models.keys())

    def _preprocess(self, image):
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        return x

    def predict(self, image, members=None, weighted=False):
        members = members or self.member_keys
        x = self._preprocess(image)
        probs_list = []
        with torch.no_grad():
            for key in members:
                logits = self.models[key](x)
                probs_list.append(torch.softmax(logits.float(), dim=1)[0].cpu().numpy())

        if weighted:
            w = np.array([self.val_f1[k] for k in members])
            w = w / w.sum()
        else:
            w = np.ones(len(members)) / len(members)

        ensemble_probs = np.tensordot(w, np.stack(probs_list, axis=0), axes=([0], [0]))
        pred_idx = int(ensemble_probs.argmax())
        return {
            "predicted_class": CLASS_NAMES[pred_idx],
            "ensemble_probs": ensemble_probs,
            "per_model_probs": dict(zip(members, probs_list)),
        }

    def predict_batch_probs(self, images, members=None, weighted=False):
        """Returns (N, 3) ensemble probability array for a list of PIL images."""
        return np.stack([self.predict(im, members, weighted)["ensemble_probs"] for im in images])
