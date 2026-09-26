"""Prediktor ensemble stacking untuk prototipe aplikasi.

Konfigurasi final yang dilaporkan pada Bab IV adalah ensemble stacking, bukan
soft-voting. Kelas ini menerapkan aturan yang sama persis dengan
stacking_ensemble.py saat pengujian: untuk tiap fold k, probabilitas
`efficientnet_b0_fold{k}` dan `resnet50_fold{k}` disatukan menjadi vektor fitur
berdimensi enam, diteruskan ke meta-learner regresi logistik, lalu hasil
kelima fold dirata-ratakan.

Dengan demikian angka yang keluar di aplikasi identik dengan angka yang
dilaporkan pada naskah, bukan sekadar mendekati.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from data.dataset import CLASS_NAMES
from inference.predictor import EnsemblePredictor, N_FOLDS

MALIGNANT_IDX = CLASS_NAMES.index("Malignant")


class StackingPredictor:
    def __init__(self, model_dir, meta_learner_path, device=None, img_size=512):
        self.base = EnsemblePredictor(model_dir, device=device, img_size=img_size)
        with open(meta_learner_path, "rb") as f:
            self.meta = pickle.load(f)
        self.member_keys = self.base.member_keys

    def predict(self, image, threshold=0.50):
        x = self.base._preprocess(image)
        import torch
        per_model = {}
        with torch.no_grad():
            for key in self.member_keys:
                logits = self.base.models[key](x)
                per_model[key] = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()

        fold_probs = []
        for k in range(N_FOLDS):
            eff, res = per_model.get(f"efficientnet_b0_fold{k}"), per_model.get(f"resnet50_fold{k}")
            if eff is None or res is None:
                continue
            feats = np.concatenate([eff, res]).reshape(1, -1)
            fold_probs.append(self.meta.predict_proba(feats)[0])
        assert fold_probs, "tidak ada pasangan fold lengkap untuk stacking"

        probs = np.mean(fold_probs, axis=0)
        # aturan ambang sama dengan stacking_ensemble.py: kelas Malignant
        # diutamakan begitu probabilitasnya mencapai ambang
        idx = MALIGNANT_IDX if probs[MALIGNANT_IDX] >= threshold else int(probs.argmax())
        return {
            "predicted_class": CLASS_NAMES[idx],
            "ensemble_probs": probs,
            "per_model_probs": per_model,
            "per_fold_probs": {f"fold{k}": p for k, p in enumerate(fold_probs)},
        }
