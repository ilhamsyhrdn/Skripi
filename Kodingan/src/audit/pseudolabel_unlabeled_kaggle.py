"""Assign labels to the Kaggle "Test cases" images that ship without any.

These 197 unique images (394 files, duplicated across two datasets) carry no
label at all, and their true label cannot be recovered: a hash search against
every labelled image found zero matches, so they are not duplicates of
anything already labelled.

The label is therefore inferred the same way the ambiguous LIDC-IDRI nodules
were handled (Zhang et al., 2022): embed every image with an ImageNet
pre-trained EfficientNet-B0, then take a majority vote over the k nearest
labelled neighbours by cosine similarity.

Because these images have no ground-truth signal whatsoever -- unlike the
LIDC nodules, which at least had radiologist scores that were merely
inconclusive -- a stricter rule is applied: a label is accepted only when at
least MIN_AGREE of the k neighbours agree. Images below that bar stay
unlabelled and are left out of training rather than guessed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import build_transforms

MANIFEST = Path("D:/skripsi/Kodingan/outputs/manifests/kaggle_full_manifest.csv")
LABELED_POOL = Path("D:/skripsi/Kodingan/outputs/manifests/kaggle_canonical_pool.csv")
OUT_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/kaggle_pseudolabeled.csv")

K = 5
MIN_AGREE = 4  # of K -- stricter than the LIDC relabeling (simple majority)


@torch.no_grad()
def embed(paths, model, tf, device, tag):
    feats = []
    for i, p in enumerate(paths, 1):
        with Image.open(p) as im:
            x = tf(im.convert("RGB")).unsqueeze(0).to(device)
        f = model(x).flatten(1)
        feats.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy()[0])
        if i % 200 == 0 or i == len(paths):
            print(f"  [{tag}] {i}/{len(paths)}", flush=True)
    return np.stack(feats)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    m = pd.read_csv(MANIFEST)
    m = m[~m["is_synthetic_source"].astype(bool)]
    unlabeled = m[~m["is_labeled"].astype(bool)].drop_duplicates(subset="md5").reset_index(drop=True)
    labeled = pd.read_csv(LABELED_POOL)

    print(f"Citra tanpa label (unik) : {len(unlabeled)}", flush=True)
    print(f"Citra berlabel (acuan)   : {len(labeled)}", flush=True)

    net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    net.classifier = torch.nn.Identity()
    net = net.to(device).eval()
    tf = build_transforms(224, train=False)

    ref = embed(labeled["path"].tolist(), net, tf, device, "berlabel")
    qry = embed(unlabeled["path"].tolist(), net, tf, device, "tanpa label")

    ref_labels = labeled["canonical_label"].to_numpy()
    sims = qry @ ref.T                      # cosine similarity (vectors are L2-normalised)
    topk = np.argsort(-sims, axis=1)[:, :K]

    rows = []
    for i, idx in enumerate(topk):
        votes = pd.Series(ref_labels[idx]).value_counts()
        best, n_best = votes.index[0], int(votes.iloc[0])
        rows.append({
            "path": unlabeled.loc[i, "path"],
            "source_dataset": unlabeled.loc[i, "source_dataset"],
            "filename": unlabeled.loc[i, "filename"],
            "md5": unlabeled.loc[i, "md5"],
            "pseudo_label": best,
            "n_setuju": n_best,
            "similarity_rata2": float(sims[i, idx].mean()),
            "diterima": n_best >= MIN_AGREE,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    acc = out[out["diterima"]]
    print("\n=== Hasil pelabelan ===", flush=True)
    print(f"Diterima (>= {MIN_AGREE}/{K} tetangga setuju): {len(acc)} dari {len(out)}", flush=True)
    print(f"Ditolak (kesepakatan lemah, tidak dipakai)   : {len(out) - len(acc)}", flush=True)
    if len(acc):
        print("\nDistribusi label yang diterima:", flush=True)
        print(acc["pseudo_label"].value_counts().to_string(), flush=True)
        print(f"\nRata-rata kemiripan: {acc['similarity_rata2'].mean():.3f}", flush=True)
    print(f"\nTersimpan: {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
