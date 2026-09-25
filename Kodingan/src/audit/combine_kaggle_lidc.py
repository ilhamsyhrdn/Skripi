"""Combine the Kaggle canonical pool (folder-label based) with the LIDC-IDRI
canonical pool (nodule-score + pathology-corrected label based) into one
unified training manifest, after a cross-dataset duplicate check (MD5 +
perceptual hash) between the two sources -- they come from completely
different acquisition pipelines (Kaggle: pre-cropped 2D photos of CT slices;
LIDC-IDRI: our own DICOM-derived 224x224 nodule crops), so no overlap is
expected, but it is verified rather than assumed.

split_group is kept source-specific (Kaggle uses case_key, LIDC-IDRI uses
PatientID) so StratifiedGroupKFold still prevents leakage within each
source; the two sources cannot share a split_group by construction.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

KAGGLE_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/kaggle_canonical_pool.csv")
LIDC_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/lidc_canonical_pool_final.csv")
OUT_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/combined_canonical_pool.csv")


def main():
    kaggle = pd.read_csv(KAGGLE_CSV)
    lidc = pd.read_csv(LIDC_CSV)

    kaggle_out = pd.DataFrame({
        "path": kaggle["path"],
        "canonical_label": kaggle["canonical_label"],
        "split_group": "KAGGLE::" + kaggle["case_key"].astype(str),
        "source_dataset": "Kaggle::" + kaggle["source_dataset"],
        "md5": kaggle["md5"],
        "phash": kaggle["phash"],
    })
    lidc_out = pd.DataFrame({
        "path": lidc["path"],
        "canonical_label": lidc["canonical_label"],
        "split_group": "LIDC::" + lidc["split_group"].astype(str),
        "source_dataset": "LIDC-IDRI",
        "md5": None,
        "phash": None,
    })

    print(f"Kaggle: {len(kaggle_out)} images", flush=True)
    print(kaggle_out["canonical_label"].value_counts(), flush=True)
    print(f"\nLIDC-IDRI: {len(lidc_out)} images", flush=True)
    print(lidc_out["canonical_label"].value_counts(), flush=True)

    # cross-dataset duplicate check: fingerprint LIDC images, compare vs Kaggle md5/phash sets
    print("\nCross-checking for accidental cross-dataset duplicates...", flush=True)
    kaggle_md5 = set(kaggle_out["md5"].dropna())
    kaggle_phash = set(kaggle_out["phash"].dropna())
    n_dupe = 0
    for p in lidc_out["path"]:
        with Image.open(p) as im:
            im_rgb = im.convert("RGB")
            ph = str(imagehash.phash(im_rgb, hash_size=8))
        with open(p, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        if md5 in kaggle_md5 or ph in kaggle_phash:
            n_dupe += 1
    print(f"Cross-dataset duplicates found: {n_dupe} (expected 0 -- different acquisition pipelines)", flush=True)

    combined = pd.concat([kaggle_out, lidc_out], ignore_index=True)
    combined = combined.drop(columns=["md5", "phash"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)

    print(f"\n=== Combined pool ===", flush=True)
    print(f"Total: {len(combined)} images, {combined['split_group'].nunique()} unique groups", flush=True)
    print(combined["canonical_label"].value_counts(), flush=True)
    print(f"\nSaved: {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
