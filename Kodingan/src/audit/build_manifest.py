"""
Dataset audit & manifest builder.

Walks D:/skripsi/Dataset (7 separately-downloaded Kaggle "lung cancer CT" datasets),
fingerprints every image (MD5 + perceptual hash), infers a canonical label from the
folder name, groups duplicate/near-duplicate images across AND within the 7 source
folders (exact re-uploads, recompressed re-uploads, Windows "- Copy" duplicates,
synthetic augmentation, and adjacent-slice near-duplicates from the same CT study),
and writes:

  1) Kodingan/outputs/manifests/full_manifest.csv       -- every image, every flag
  2) Kodingan/outputs/manifests/canonical_pool.csv       -- de-duplicated, labeled,
     leakage-safe pool used for train/val/test splitting
  3) Kodingan/outputs/manifests/duplicate_groups.csv     -- every group with >1 member
  4) Kejanggalan Dataset/*.md                            -- human-readable findings

Run:  Kodingan/.venv/Scripts/python.exe Kodingan/src/audit/build_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO_ROOT / "Dataset"
OUT_DIR = REPO_ROOT / "Kodingan" / "outputs" / "manifests"
DOCS_DIR = REPO_ROOT / "Kejanggalan Dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Datasets that are synthetic augmentations of another dataset already present here.
# Excluded from the canonical training pool by policy (see 02_augmented_dataset_overlap.md).
SYNTHETIC_SOURCES = {"IQ-OTHNCCD Lung Cancer Dataset (Augmented) (Subhajeet Das)"}

# Near-duplicate perceptual-hash Hamming-distance thresholds (64-bit phash).
PHASH_EXACT_BITS = 0
NEARDUP_THRESHOLD_ADJACENT = 6    # within-folder, numerically adjacent case indices (auto-merged)
NEARDUP_THRESHOLD_GLOBAL = 6      # cross-source, bucketed comparison (candidates only, not merged)


# --------------------------------------------------------------------------- #
# 1. Discovery + label inference
# --------------------------------------------------------------------------- #

def infer_label(path_parts: list[str]) -> tuple[str, str, bool]:
    """Return (canonical_label, subtype, is_labeled) from the folder path."""
    joined = " / ".join(path_parts).lower()

    if "test cases" in joined and not any(
        k in joined for k in ["adenocarcinoma", "squamous", "large", "bengin", "benign", "malignant", "normal"]
    ):
        return "Unlabeled", "Unlabeled", False

    if "adenocarcinoma" in joined:
        return "Malignant", "Adenocarcinoma", True
    if "large.cell" in joined or "large cell" in joined:
        return "Malignant", "LargeCellCarcinoma", True
    if "squamous" in joined:
        return "Malignant", "SquamousCellCarcinoma", True
    if "malignant" in joined:
        return "Malignant", "Unspecified", True
    if "bengin" in joined or "benign" in joined:
        return "Benign", "Benign", True
    if "normal" in joined:
        return "Normal", "Normal", True
    return "Unlabeled", "Unlabeled", False


def discover_images() -> pd.DataFrame:
    records = []
    for source_dir in sorted(p for p in DATASET_ROOT.iterdir() if p.is_dir()):
        source_name = source_dir.name
        for fp in source_dir.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in IMG_EXTS:
                rel = fp.relative_to(source_dir)
                folder_parts = list(rel.parts[:-1])
                label, subtype, is_labeled = infer_label([source_name] + folder_parts)
                records.append(
                    {
                        "path": str(fp),
                        "source_dataset": source_name,
                        "raw_folder": "/".join(folder_parts) if folder_parts else "(root)",
                        "filename": fp.name,
                        "canonical_label": label,
                        "subtype": subtype,
                        "is_labeled": is_labeled,
                        "is_synthetic_source": source_name in SYNTHETIC_SOURCES,
                    }
                )
    df = pd.DataFrame.from_records(records)
    return df


# --------------------------------------------------------------------------- #
# 2. Case-key extraction (within-source near-duplicate-slice safety net)
# --------------------------------------------------------------------------- #

_COPY_RE = re.compile(r"\s*-\s*copy(\s*\(\d+\))?", re.IGNORECASE)
_LEADING_NUM_RE = re.compile(r"^(\d+)")
_PAREN_NUM_RE = re.compile(r"\((\d+)\)")


def extract_case_key(filename: str, source_dataset: str, raw_folder: str) -> str:
    stem = re.sub(r"\.(jpg|jpeg|png|bmp)$", "", filename, flags=re.IGNORECASE)
    stem = _COPY_RE.sub("", stem).strip()
    m = _LEADING_NUM_RE.match(stem)
    if m:
        num = m.group(1)
    else:
        m2 = _PAREN_NUM_RE.search(stem)
        num = m2.group(1) if m2 else stem.lower()
    return f"{source_dataset}|{raw_folder}::{num}"


# --------------------------------------------------------------------------- #
# 3. Fingerprinting
# --------------------------------------------------------------------------- #

def fingerprint(path: str) -> dict:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im = im.convert("RGB")
            width, height = im.size
            mode_orig = im.mode
            ph = imagehash.phash(im, hash_size=8)
            ah = imagehash.average_hash(im, hash_size=8)
        with open(path, "rb") as fbin:
            md5 = hashlib.md5(fbin.read()).hexdigest()
        return {
            "corrupt": False,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 3) if height else None,
            "phash": str(ph),
            "phash_int": int(str(ph), 16),
            "ahash": str(ah),
            "md5": md5,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "corrupt": True,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "phash": None,
            "phash_int": None,
            "ahash": None,
            "md5": None,
            "error": str(exc),
        }


# --------------------------------------------------------------------------- #
# 4. Union-Find
# --------------------------------------------------------------------------- #

class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    t0 = time.time()
    print(f"[1/6] Discovering images under {DATASET_ROOT} ...", flush=True)
    df = discover_images()
    print(f"      found {len(df):,} image files across {df['source_dataset'].nunique()} source datasets", flush=True)

    cache_path = OUT_DIR / "_fingerprint_cache.csv"
    if cache_path.exists():
        print(f"[2/6] Loading cached fingerprints from {cache_path.name} ...", flush=True)
        cache = pd.read_csv(cache_path)
        df = df.merge(cache, on="path", how="left")
        missing = df["md5"].isna() & (~df.get("corrupt", pd.Series(False, index=df.index)).astype(bool))
        if missing.any():
            print(f"      {int(missing.sum())} new files not in cache, fingerprinting those ...", flush=True)
            new_records = [fingerprint(p) for p in tqdm(df.loc[missing, "path"], ncols=100)]
            for col_i, idx in enumerate(df.index[missing]):
                for k, v in new_records[col_i].items():
                    df.at[idx, k] = v
    else:
        print("[2/6] Fingerprinting (MD5 + perceptual hash) -- this reads every pixel once ...", flush=True)
        records = []
        for path in tqdm(df["path"].tolist(), ncols=100):
            records.append(fingerprint(path))
        fp_df = pd.DataFrame(records)
        df = pd.concat([df.reset_index(drop=True), fp_df.reset_index(drop=True)], axis=1)
        df[["path", "corrupt", "width", "height", "aspect_ratio", "phash", "phash_int", "ahash", "md5"]].to_csv(
            cache_path, index=False
        )

    n_corrupt = int(df["corrupt"].sum())
    print(f"      corrupt/unreadable files: {n_corrupt}", flush=True)

    df["case_key"] = [
        extract_case_key(fn, src, rf)
        for fn, src, rf in zip(df["filename"], df["source_dataset"], df["raw_folder"])
    ]

    # The "Augmented" (Das) source is EXCLUDED from the duplicate-grouping / union-find pool
    # entirely, not just from the final canonical pool. Letting its ~3.6k flip/rotate variants
    # take part in exact-phash matching caused false bridges between unrelated real images (a
    # flipped Benign crop and a flipped Normal crop can coincidentally collide on a 64-bit phash
    # when the underlying content is low-detail) that transitively merged large, differently
    # labeled chunks of the REAL datasets into one giant "conflicting label" blob and made the
    # entire Benign class disappear from the canonical pool. See
    # Kejanggalan Dataset/02_augmented_dataset_overlap.md for the full account and the
    # (separately computed, non-grouping) overlap evidence against Das's dataset.
    not_corrupt = df[~df["corrupt"]].copy().reset_index(drop=True)
    synthetic = not_corrupt[not_corrupt["is_synthetic_source"]].copy().reset_index(drop=True)
    valid = not_corrupt[~not_corrupt["is_synthetic_source"]].copy().reset_index(drop=True)
    n = len(valid)
    dsu = DSU(n)

    print("[3/6] Grouping exact duplicates (MD5) ...", flush=True)
    md5_groups: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(valid["md5"]):
        md5_groups[m].append(i)
    n_exact_md5_groups = sum(1 for v in md5_groups.values() if len(v) > 1)
    for idxs in md5_groups.values():
        for j in idxs[1:]:
            dsu.union(idxs[0], j)
    print(f"      exact byte-identical groups (>1 file): {n_exact_md5_groups}", flush=True)

    print("[4/6] Grouping exact perceptual-hash matches (catches re-saved/re-compressed re-uploads) ...", flush=True)
    ph_groups: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(valid["phash_int"]):
        ph_groups[p].append(i)
    n_exact_ph_groups = sum(1 for v in ph_groups.values() if len(v) > 1)
    for idxs in ph_groups.values():
        for j in idxs[1:]:
            dsu.union(idxs[0], j)
    print(f"      exact phash groups (>1 file): {n_exact_ph_groups}", flush=True)

    print("[5/6] Near-duplicate passes ...", flush=True)
    # (a) SAME case_key within the same (source_dataset, raw_folder): this targets the exact
    # pattern observed on disk -- "10.png" / "10 (2).png" / "10 - Copy.png" (Windows copy-paste
    # duplicates) and "000005 (3).png" / "000005 (9).png" (same patient ID, different crop/slice
    # export) all reduce to the identical case_key. This is a filename-evidence-based merge, not
    # a perceptual-similarity guess, so it does not suffer from the low-entropy-CT-domain false
    # positives that a plain "numerically adjacent" + loose phash-distance rule produced in an
    # earlier version of this script (it collapsed genuinely different patients together --
    # Benign fell from ~480 raw images to an implausible 24 canonical groups; see
    # Kejanggalan Dataset/00_methodology_note.md for the full account).
    by_key: dict[str, list[int]] = defaultdict(list)
    for i, ck in enumerate(valid["case_key"]):
        by_key[ck].append(i)
    adj_unions = 0
    for idxs in by_key.values():
        if len(idxs) < 2:
            continue
        for j in idxs[1:]:
            dsu.union(idxs[0], j)
        adj_unions += len(idxs) - 1
    print(f"      same-case_key (filename-evidence) unions: {adj_unions}", flush=True)

    # (b) cross-source near-duplicates: bucket by top 24 bits of phash within each canonical_label.
    #
    # IMPORTANT: lung CT slices are visually low-entropy (same circular FOV, same lung-window
    # contrast, similar rib/spine layout) so a loose phash distance is NOT a safe cross-source
    # duplicate signal here -- it also fires on genuinely different patients. A first pass with
    # distance<=6 on phash alone produced ~80k unions and collapsed the Benign class from ~480
    # raw images down to 24 canonical groups, which is not plausible (the source datasets only
    # claim ~120 unique benign patients between them, not 24) -- classic over-merging.
    #
    # Cross-source pairs are therefore only ever *reported* as review candidates (two independent
    # hashes -- phash AND ahash -- must both agree at a tight distance), never auto-merged into
    # the canonical grouping. Only exact MD5 / exact phash / strict adjacent-slice-within-folder
    # matches are trusted enough to merge groups automatically.
    candidate_rows = []
    for label, sub in valid.groupby("canonical_label"):
        buckets: dict[int, list[int]] = defaultdict(list)
        for idx, ph in zip(sub.index, sub["phash_int"]):
            buckets[ph >> 40].append(idx)
        for idxs in tqdm(buckets.values(), ncols=100, desc=f"cross-source bucket check [{label}]", leave=False):
            if len(idxs) < 2:
                continue
            for bi in range(len(idxs)):
                for bj in range(bi + 1, len(idxs)):
                    a, b = idxs[bi], idxs[bj]
                    if valid.at[a, "source_dataset"] == valid.at[b, "source_dataset"]:
                        continue  # within-source already handled above
                    dph = hamming(valid.at[a, "phash_int"], valid.at[b, "phash_int"])
                    if dph > NEARDUP_THRESHOLD_GLOBAL:
                        continue
                    dah = hamming(
                        int(valid.at[a, "ahash"], 16), int(valid.at[b, "ahash"], 16)
                    )
                    if dah <= NEARDUP_THRESHOLD_GLOBAL:
                        candidate_rows.append(
                            {
                                "canonical_label": label,
                                "path_a": valid.at[a, "path"],
                                "source_a": valid.at[a, "source_dataset"],
                                "path_b": valid.at[b, "path"],
                                "source_b": valid.at[b, "source_dataset"],
                                "phash_distance": dph,
                                "ahash_distance": dah,
                            }
                        )
    candidates_df = pd.DataFrame(candidate_rows)
    candidates_df.to_csv(OUT_DIR / "cross_source_near_duplicate_candidates.csv", index=False)
    global_unions = 0
    print(
        f"      cross-source near-duplicate CANDIDATES logged for review "
        f"(not auto-merged): {len(candidates_df)}",
        flush=True,
    )

    print("[6/6] Finalizing groups + labels ...", flush=True)
    roots = [dsu.find(i) for i in range(n)]
    valid["group_root"] = roots
    root_to_gid = {r: gid for gid, r in enumerate(sorted(set(roots)))}
    valid["group_id"] = valid["group_root"].map(root_to_gid)

    group_sizes = valid.groupby("group_id").size()
    valid["group_size"] = valid["group_id"].map(group_sizes)

    # label conflict detection per group (ignore Unlabeled members)
    def label_conflict(sub: pd.DataFrame) -> bool:
        labs = set(sub.loc[sub["is_labeled"], "canonical_label"])
        return len(labs) > 1

    conflicts = valid.groupby("group_id").apply(label_conflict)
    valid["label_conflict"] = valid["group_id"].map(conflicts)

    def group_sources(sub: pd.DataFrame) -> str:
        return "|".join(sorted(sub["source_dataset"].unique()))

    src_per_group = valid.groupby("group_id").apply(group_sources)
    valid["group_sources"] = valid["group_id"].map(src_per_group)
    valid["cross_dataset_duplicate"] = valid["group_sources"].str.contains(r"\|")

    corrupt_df = df[df["corrupt"]].copy()
    corrupt_df["group_id"] = -1
    corrupt_df["group_size"] = 1
    corrupt_df["label_conflict"] = False
    corrupt_df["group_sources"] = corrupt_df["source_dataset"]
    corrupt_df["cross_dataset_duplicate"] = False

    # Synthetic (Das "Augmented") rows never join the real union-find groups. We still record,
    # for each one, whether it exact-matches (MD5 or phash) something already in the real pool --
    # this is pure evidence for the audit report, it does not feed back into `valid`'s grouping.
    real_md5_set = set(valid["md5"])
    real_phash_set = set(valid["phash"])
    synthetic["group_id"] = -3
    synthetic["group_size"] = 1
    synthetic["label_conflict"] = False
    synthetic["group_sources"] = synthetic["source_dataset"]
    synthetic["cross_dataset_duplicate"] = False
    synthetic["matches_real_pool_md5"] = synthetic["md5"].isin(real_md5_set)
    synthetic["matches_real_pool_phash"] = synthetic["phash"].isin(real_phash_set)
    n_synth_exact = int(synthetic["matches_real_pool_md5"].sum())
    n_synth_phash = int((synthetic["matches_real_pool_phash"] & ~synthetic["matches_real_pool_md5"]).sum())
    print(
        f"      synthetic-source (Das) overlap vs real pool: {n_synth_exact} byte-identical, "
        f"+{n_synth_phash} more exact-phash-identical (out of {len(synthetic)})",
        flush=True,
    )

    full = pd.concat([valid, synthetic, corrupt_df], ignore_index=True, sort=False)
    full.to_csv(OUT_DIR / "full_manifest.csv", index=False)

    dup_groups = valid[valid["group_size"] > 1].sort_values(["group_size", "group_id"], ascending=[False, True])
    dup_groups.to_csv(OUT_DIR / "duplicate_groups.csv", index=False)

    # ------------------------------------------------------------------- #
    # Canonical pool: one representative per group, labeled, non-conflicting,
    # excluding synthetic-augmentation source entirely.
    # ------------------------------------------------------------------- #
    eligible = valid[
        valid["is_labeled"]
        & (~valid["is_synthetic_source"])
        & (~valid["label_conflict"])
    ].copy()

    def pick_representative(sub: pd.DataFrame) -> pd.Series:
        # Prefer the highest-resolution, most "original"-looking file as representative.
        return sub.sort_values(["width"], ascending=False).iloc[0]

    canonical = eligible.groupby("group_id", as_index=False).apply(pick_representative).reset_index(drop=True)
    canonical.to_csv(OUT_DIR / "canonical_pool.csv", index=False)

    # ------------------------------------------------------------------- #
    # Summary stats for the report
    # ------------------------------------------------------------------- #
    summary = {
        "total_files_found": int(len(df)),
        "corrupt_files": n_corrupt,
        "sources": sorted(df["source_dataset"].unique().tolist()),
        "raw_label_counts_by_source": (
            df.groupby(["source_dataset", "canonical_label"]).size().unstack(fill_value=0).to_dict("index")
        ),
        "exact_md5_duplicate_groups": n_exact_md5_groups,
        "exact_phash_duplicate_groups": n_exact_ph_groups,
        "adjacent_slice_near_duplicate_unions": adj_unions,
        "cross_source_near_duplicate_unions": global_unions,
        "total_groups_all_valid_images": int(valid["group_id"].nunique()),
        "groups_with_gt1_member": int((group_sizes > 1).sum()),
        "cross_dataset_duplicate_groups": int(
            valid.loc[valid["cross_dataset_duplicate"], "group_id"].nunique()
        ),
        "label_conflict_groups": int(valid.loc[valid["label_conflict"], "group_id"].nunique()),
        "synthetic_source_images_excluded": int(len(synthetic)),
        "synthetic_images_exact_byte_match_to_real_pool": n_synth_exact,
        "synthetic_images_exact_phash_match_to_real_pool": n_synth_phash,
        "unlabeled_images_excluded": int((~valid["is_labeled"]).sum()),
        "canonical_pool_size": int(len(canonical)),
        "canonical_pool_label_counts": canonical["canonical_label"].value_counts().to_dict(),
        "canonical_pool_subtype_counts": canonical["subtype"].value_counts().to_dict(),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n================ SUMMARY ================")
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:4000])
    print(f"\nDone in {summary['elapsed_seconds']}s. Manifests written to {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
