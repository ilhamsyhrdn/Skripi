"""Loads the trained ensemble (cached) and exposes a single `predict()` call used by both
the Streamlit app and any future CLI/API. Kept framework-light on purpose so Streamlit's
`st.cache_resource` can wrap `EnsemblePredictor` directly."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.dataset import CLASS_NAMES, build_transforms
from src.models.factory import build_model, get_last_conv_layer
from src.inference.gradcam import GradCAM

ARCHS = ["efficientnet_b0", "resnet50"]
N_FOLDS = 5


class EnsemblePredictor:
    def __init__(self, model_dir: Path, device: str | None = None, archs: list[str] | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.archs = archs or ARCHS
        self.models: dict[str, torch.nn.Module] = {}
        self.val_f1: dict[str, float] = {}
        self.transform = build_transforms(224, train=False)

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

    @property
    def member_keys(self) -> list[str]:
        return list(self.models.keys())

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image, members: list[str] | None = None, weighted: bool = False) -> dict:
        members = members or self.member_keys
        x = self._preprocess(image)
        probs_list = []
        per_model = {}
        with torch.no_grad():
            for key in members:
                logits = self.models[key](x)
                p = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
                probs_list.append(p)
                per_model[key] = {CLASS_NAMES[i]: float(p[i]) for i in range(len(CLASS_NAMES))}

        if weighted:
            w = np.array([self.val_f1[k] for k in members])
            w = w / w.sum()
        else:
            w = np.ones(len(members)) / len(members)
        ensemble_probs = np.tensordot(w, np.stack(probs_list, axis=0), axes=([0], [0]))
        pred_idx = int(ensemble_probs.argmax())

        return {
            "predicted_class": CLASS_NAMES[pred_idx],
            "predicted_idx": pred_idx,
            "ensemble_probs": {CLASS_NAMES[i]: float(ensemble_probs[i]) for i in range(len(CLASS_NAMES))},
            "per_model_probs": per_model,
            "members_used": members,
        }

    def gradcam(self, image: Image.Image, arch: str = "efficientnet_b0", fold: int = 0, class_idx: int | None = None):
        key = f"{arch}_fold{fold}"
        model = self.models[key]
        x = self._preprocess(image)
        x.requires_grad_(False)
        target_layer = get_last_conv_layer(model, arch)
        cam_engine = GradCAM(model, target_layer)
        cam, used_class_idx, probs = cam_engine(x, class_idx=class_idx)
        return cam, CLASS_NAMES[used_class_idx], {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
