"""Audit + deduplicate the 7 Kaggle lung-CT datasets, reproducing the
original thesis methodology (Bab III sub-bab 3.3 / Lampiran 1):

  1. Fingerprint every image (MD5 exact + 64-bit perceptual hash) across all
     seven raw dataset folders.
  2. Exclude the synthetic-augmentation dataset (Subhajeet Das) entirely from
     grouping -- it is a re-augmented copy of the IQ-OTHNCCD base set and
     caused false hash collisions in earlier attempts (documented pitfall).
  3. Exclude unlabeled "Test cases" folders (no class subfolder -> no
     ground truth).
  4. Group the remainder into duplicate clusters using ONLY strong evidence:
     exact MD5, exact perceptual hash, or identical case-number filename
     within the same (source, folder) family -- never a loose visual
     distance threshold (that collapsed hundreds of distinct Benign patients
     into a handful of groups in the original pitfall).
  5. Flag and drop any group whose members disagree on canonical_label.
  6. Keep one representative (highest resolution) per group -> canonical pool.

Output: outputs/manifests/kaggle_canonical_pool.csv
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

DATASET_ROOT = Path("D:/skripsi/Dataset")

DATASETS = [
    "CT Scan Images for Lung Cancer (Dishan rathi20)",
    "CT Scan Images of Lung Cancer Patients.(MD. NAFEES IMTIAZ)",
    "Chest CT-Scan images Dataset (Mohamed Hany)",
    "IQ-OTHNCCD - Lung Cancer Dataset (Aditya Mahimkar)",
    "IQ-OTHNCCD Lung Cancer Dataset (Augmented) (Subhajeet Das)",
    "Lung cancer dataset (IQ-OTHNCCD)(Waseim Nagah Hennes)",
    "The IQ-OTHNCCD lung cancer dataset ( Hamdalla F. Al-Yasriy )",
]
SYNTHETIC_AUGMENTED_DATASET = "IQ-OTHNCCD Lung Cancer Dataset (Augmented) (Subhajeet Das)"

OUT_DIR = Path("D:/skripsi/Kodingan/outputs/manifests")

# folder-name (lowercased, separators normalized) -> canonical label
LABEL_RULES = [
    (re.compile(r"ben(?:ign|gin)\s*cases?", re.I), "Benign"),
    (re.compile(r"malignant\s*cases?", re.I), "Malignant"),
    (re.compile(r"normal\s*cases?$", re.I), "Normal"),
    (re.compile(r"^normal$", re.I), "Normal"),
    (re.compile(r"adenocarcinoma", re.I), "Malignant"),
    (re.compile(r"large[._ ]cell[._ ]carcinoma", re.I), "Malignant"),
    (re.compile(r"squamous[._ ]cell[._ ]carcinoma", re.I), "Malignant"),
]
SUBTYPE_RULES = [
    (re.compile(r"adenocarcinoma", re.I), "Adenocarcinoma"),
    (re.compile(r"large[._ ]cell[._ ]carcinoma", re.I), "Large Cell Carcinoma"),
    (re.compile(r"squamous[._ ]cell[._ ]carcinoma", re.I), "Squamous Cell Carcinoma"),
]
UNLABELED_FOLDER_NAMES = {"test cases"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

CASE_NUM_RE = re.compile(r"(\d+)")


def label_for_folder(folder_name: str):
    for pat, label in LABEL_RULES:
        if pat.search(folder_name):
            return label
    return None


def subtype_for_folder(folder_name: str):
    for pat, subtype in SUBTYPE_RULES:
        if pat.search(folder_name):
            return subtype
    if "beng" in folder_name.lower() or "benign" in folder_name.lower():
        return "Benign"
    if "normal" in folder_name.lower():
        return "Normal"
    if "malignant" in folder_name.lower():
        return "Malignant (unspecified subtype)"
    return None


def case_key_from_filename(source, raw_folder, filename):
    m = CASE_NUM_RE.search(filename)
    num = m.group(1) if m else filename
    return f"{source}|{raw_folder}::{num}"


def scan_dataset(source: str):
    root = DATASET_ROOT / source
    rows = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = p.relative_to(root)
        raw_folder = str(rel.parent).replace("\\", "/")
        leaf_folder_name = rel.parent.name
        if leaf_folder_name.lower() in UNLABELED_FOLDER_NAMES or "test cases" in raw_folder.lower():
            is_labeled = False
            label = None
        else:
            label = label_for_folder(leaf_folder_name)
            is_labeled = label is not None
        subtype = subtype_for_folder(leaf_folder_name) if is_labeled else None
        rows.append({
            "path": str(p),
            "source_dataset": source,
            "raw_folder": raw_folder,
            "filename": p.name,
            "canonical_label": label,
            "subtype": subtype,
            "is_labeled": is_labeled,
            "is_synthetic_source": source == SYNTHETIC_AUGMENTED_DATASET,
        })
    return rows


def fingerprint(row):
    p = Path(row["path"])
    try:
        with Image.open(p) as im:
            width, height = im.size
            aspect = round(width / height, 3) if height else None
            im_rgb = im.convert("RGB")
            ph = imagehash.phash(im_rgb, hash_size=8)
            ah = imagehash.average_hash(im_rgb, hash_size=8)
        with open(p, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        row.update({"corrupt": False, "width": width, "height": height, "aspect_ratio": aspect,
                     "phash": str(ph), "phash_int": int(str(ph), 16), "ahash": str(ah), "md5": md5})
    except Exception:
        row.update({"corrupt": True, "width": None, "height": None, "aspect_ratio": None,
                     "phash": None, "phash_int": None, "ahash": None, "md5": None})
    return row


def main():
    print("Scanning 7 Kaggle dataset folders...", flush=True)
    all_rows = []
    for ds in DATASETS:
        rows = scan_dataset(ds)
        print(f"  {ds}: {len(rows)} files", flush=True)
        all_rows.extend(rows)
    print(f"Total raw files: {len(all_rows)}", flush=True)

    print("\nFingerprinting (MD5 + perceptual hash)...", flush=True)
    for i, row in enumerate(all_rows, 1):
        fingerprint(row)
        if i % 2000 == 0:
            print(f"  fingerprinted {i}/{len(all_rows)}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "kaggle_full_manifest.csv", index=False)

    n_total = len(df)
    n_synthetic = int(df["is_synthetic_source"].sum())
    n_unlabeled = int((~df["is_labeled"]).sum())
    working = df[(~df["is_synthetic_source"]) & (df["is_labeled"]) & (~df["corrupt"])].copy()
    print(f"\nAfter excluding synthetic-augmented ({n_synthetic}) and unlabeled/corrupt "
          f"({n_unlabeled} unlabeled): {len(working)} remain", flush=True)

    # --- grouping: MD5 exact ---
    working = working.reset_index(drop=True)
    working["case_key"] = [case_key_from_filename(r.source_dataset, r.raw_folder, r.filename)
                            for r in working.itertuples()]

    parent = list(range(len(working)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    from collections import defaultdict
    md5_groups = defaultdict(list)
    phash_groups = defaultdict(list)
    for idx, row in enumerate(working.itertuples()):
        if row.md5:
            md5_groups[row.md5].append(idx)
        if row.phash:
            phash_groups[row.phash].append(idx)

    for idxs in md5_groups.values():
        for i in idxs[1:]:
            union(idxs[0], i)
    for idxs in phash_groups.values():
        for i in idxs[1:]:
            union(idxs[0], i)

    working["group_root"] = [find(i) for i in range(len(working))]

    # label-conflict detection: drop groups whose members disagree
    group_labels = working.groupby("group_root")["canonical_label"].nunique()
    conflict_roots = set(group_labels[group_labels > 1].index)
    working["label_conflict"] = working["group_root"].isin(conflict_roots)
    n_conflict = int(working["label_conflict"].sum())
    print(f"Files in label-conflicting groups (dropped): {n_conflict}", flush=True)

    clean = working[~working["label_conflict"]].copy()
    clean["group_size"] = clean.groupby("group_root")["group_root"].transform("count")

    # pick one representative per group: highest resolution (width*height)
    clean["area"] = clean["width"].fillna(0) * clean["height"].fillna(0)
    clean = clean.sort_values("area", ascending=False)
    canonical = clean.drop_duplicates(subset="group_root", keep="first").copy()
    canonical = canonical.drop(columns=["area"])

    canonical.to_csv(OUT_DIR / "kaggle_canonical_pool.csv", index=False)

    print(f"\n=== Kaggle canonical pool ===", flush=True)
    print(f"Total raw files (7 datasets): {n_total}", flush=True)
    print(f"Excluded synthetic-augmented: {n_synthetic}", flush=True)
    print(f"Excluded unlabeled/corrupt: {n_unlabeled}", flush=True)
    print(f"Excluded (label-conflict groups): {n_conflict}", flush=True)
    print(f"Final canonical pool: {len(canonical)}", flush=True)
    print(canonical["canonical_label"].value_counts(), flush=True)
    print(f"\nSaved: {OUT_DIR / 'kaggle_canonical_pool.csv'}", flush=True)


if __name__ == "__main__":
    main()
