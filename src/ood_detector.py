"""Deteksi out-of-distribution (OOD): menolak citra yang bukan CT paru-paru.

Pendekatan: ekstrak fitur penultimate (sebelum classifier head) dari model
terlatih, lalu ukur jarak k-nearest-neighbor citra baru terhadap seluruh
embedding training. Citra yang benar-benar CT paru akan punya tetangga dekat
di ruang fitur (karena model dilatih khusus untuk merepresentasikan citra
semacam itu); citra yang sama sekali berbeda (foto biasa, organ lain, noise)
akan berjarak jauh dari seluruh training set.

Kenapa k-NN, bukan Mahalanobis/Gaussian: dimensi fitur EfficientNet-B0 (1280)
lebih besar dari jumlah sampel training per kelas, sehingga estimasi
covariance tidak stabil. Jarak k-NN tidak butuh estimasi covariance dan lebih
robust untuk kasus n_samples yang terbatas seperti ini.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Ekstrak fitur penultimate (pre-logits) untuk semua citra di loader."""
    model.eval()
    all_embeddings = []
    for images, _ in loader:
        images = images.to(device)
        # timm models expose forward_features + forward_head(pre_logits=True)
        features = model.forward_features(images)
        pooled = model.forward_head(features, pre_logits=True)
        all_embeddings.append(pooled.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def fit_ood_detector(train_embeddings, k=5):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(train_embeddings)
    return nn


def ood_scores(detector, embeddings):
    """Rata-rata jarak ke k tetangga terdekat di training set. Skor tinggi = makin OOD."""
    distances, _ = detector.kneighbors(embeddings)
    return distances.mean(axis=1)


def calibrate_threshold(detector, val_embeddings, percentile=99.0):
    """Threshold = persentil ke-`percentile` dari skor OOD pada data in-distribution (val set).

    Dengan percentile=99, kita menerima false-rejection rate ~1% pada citra CT
    paru asli (val set) demi menolak citra yang benar-benar bukan CT paru.
    """
    scores = ood_scores(detector, val_embeddings)
    return float(np.percentile(scores, percentile))


def save_ood_detector(detector, threshold, k, path: Path):
    with open(path, "wb") as f:
        pickle.dump({"detector": detector, "threshold": threshold, "k": k}, f)


def load_ood_detector(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["detector"], data["threshold"], data["k"]
