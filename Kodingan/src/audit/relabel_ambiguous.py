"""Relabel the 226 ambiguous (malignancy score == 3) LIDC-IDRI series instead
of discarding them, following the nearest-neighbour label-transfer idea in:

  Zhang, H., Gu, X., Zhang, M., Yu, W., Chen, L., Wang, Z., Yao, F., Gu, Y.,
  & Yang, G.-Z. (2022). Re-thinking and Re-labeling LIDC-IDRI for Robust
  Pulmonary Cancer Prediction. MICCAI 2022 Workshops (LNCS 13591), Springer.
  https://doi.org/10.1007/978-3-031-16760-7_5 (arXiv:2207.14238)

Method (simplified adaptation, documented as such in Bab III): each ambiguous
nodule crop is embedded with an ImageNet-pretrained EfficientNet-B0 (feature
extractor only, no fine-tuning -- the same network family used for the main
classifier), then assigned the MAJORITY label of its k=5 nearest neighbours
among the reliably-labelled (Malignant/Benign, non-ambiguous) nodule crops in
that same embedding space. Normal crops are excluded from the neighbour pool
because the technique is specifically about resolving nodule malignancy
disagreement, not "is there a nodule at all".
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import build_transforms
from models.factory import build_model

MANIFEST = Path("D:/skripsi/Kodingan/outputs/manifests/lidc_canonical_pool_crop.csv")
OUT_MANIFEST = Path("D:/skripsi/Kodingan/outputs/manifests/lidc_canonical_pool_crop_relabeled.csv")
K_NEIGHBORS = 5


@torch.no_grad()
def embed_images(paths, device):
    model = build_model("efficientnet_b0").to(device)
    model.eval()
    feature_extractor = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten())
    transform = build_transforms(train=False)

    embeddings = []
    batch, batch_size = [], 32
    for p in paths:
        with Image.open(p) as im:
            batch.append(transform(im.convert("RGB")))
        if len(batch) == batch_size:
            x = torch.stack(batch).to(device)
            embeddings.append(feature_extractor(x).cpu().numpy())
            batch = []
    if batch:
        x = torch.stack(batch).to(device)
        embeddings.append(feature_extractor(x).cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(MANIFEST)

    reliable = df[df["canonical_label"].isin(["Malignant", "Benign"])].reset_index(drop=True)
    ambiguous = df[df["canonical_label"] == "Excluded_TBD"].reset_index(drop=True)
    print(f"Reliable anchors: {len(reliable)} ({dict(reliable['canonical_label'].value_counts())})", flush=True)
    print(f"Ambiguous to relabel: {len(ambiguous)}", flush=True)

    print("Embedding reliable anchors...", flush=True)
    reliable_emb = embed_images(reliable["path"].tolist(), device)
    print("Embedding ambiguous nodules...", flush=True)
    ambiguous_emb = embed_images(ambiguous["path"].tolist(), device)

    # cosine similarity via normalized dot product
    reliable_norm = reliable_emb / (np.linalg.norm(reliable_emb, axis=1, keepdims=True) + 1e-8)
    ambiguous_norm = ambiguous_emb / (np.linalg.norm(ambiguous_emb, axis=1, keepdims=True) + 1e-8)
    sims = ambiguous_norm @ reliable_norm.T  # (n_ambiguous, n_reliable)

    reliable_labels = reliable["canonical_label"].values
    new_labels, new_paths = [], []
    for i in range(len(ambiguous)):
        top_k_idx = np.argsort(-sims[i])[:K_NEIGHBORS]
        votes = reliable_labels[top_k_idx]
        vals, counts = np.unique(votes, return_counts=True)
        new_label = vals[np.argmax(counts)]
        new_labels.append(new_label)

        old_path = Path(ambiguous.loc[i, "path"])
        new_path = old_path.parent.parent / new_label / old_path.name
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(new_path))
        new_paths.append(str(new_path))

        if (i + 1) % 50 == 0 or i == len(ambiguous) - 1:
            print(f"  relabeled {i+1}/{len(ambiguous)}", flush=True)

    ambiguous["canonical_label"] = new_labels
    ambiguous["path"] = new_paths
    ambiguous["relabeled_from_ambiguous"] = True
    reliable["relabeled_from_ambiguous"] = False

    normal = df[df["canonical_label"] == "Normal"].copy()
    normal["relabeled_from_ambiguous"] = False

    final = pd.concat([reliable, ambiguous, normal], ignore_index=True)
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT_MANIFEST, index=False)

    print("\n=== Relabeling result ===", flush=True)
    print(pd.Series(new_labels).value_counts(), flush=True)
    print("\n=== Final full pool (all series, none discarded) ===", flush=True)
    print(final["canonical_label"].value_counts(), flush=True)
    print(f"Total: {len(final)} (was 1082 before relabeling, +{len(ambiguous)} recovered)", flush=True)
    print(f"Saved: {OUT_MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
