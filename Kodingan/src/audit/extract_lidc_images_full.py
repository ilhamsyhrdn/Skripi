"""Extract whole-slice (uncropped) LIDC-IDRI images using the FINAL labels.

This mirrors extract_lidc_images_crop.py exactly except that the slice is
saved in full instead of cropped to 224x224 around the nodule, so the two
pools differ only in framing and can be compared fairly.

Slice selection is identical to the crop pool: the slice carrying the worst
consensus nodule for Malignant/Benign series, the middle slice for Normal
series. Labels come from lidc_canonical_pool_final.csv (after ambiguous-nodule
relabeling and TCIA pathology correction).

Whole-slice PNGs produced by the earlier run are reused when present: the
pixel content of a series does not depend on its label, only the output
folder does.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

RAW_ROOT = Path("D:/skripsi/Dataset/lidc-idri")
FINAL_POOL = Path("D:/skripsi/Kodingan/outputs/manifests/lidc_canonical_pool_final.csv")
SERIES_LABELS = Path("D:/skripsi/Kodingan/outputs/manifests/lidc_series_labels.csv")
OLD_FULL_ROOT = Path("D:/skripsi/Dataset/lidc-idri-images")
OUT_IMG_ROOT = Path("D:/skripsi/Dataset/lidc-idri-images-full")
OUT_MANIFEST = Path("D:/skripsi/Kodingan/outputs/manifests/lidc_canonical_pool_full.csv")

WINDOW_LEVEL = -600
WINDOW_WIDTH = 1500


def hu_to_uint8(pixel_array, slope, intercept):
    hu = pixel_array.astype(np.float32) * slope + intercept
    lo = WINDOW_LEVEL - WINDOW_WIDTH / 2
    hi = WINDOW_LEVEL + WINDOW_WIDTH / 2
    hu = np.clip(hu, lo, hi)
    return ((hu - lo) / (hi - lo) * 255.0).astype(np.uint8)


def save_slice(ds, out_path: Path):
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    img8 = hu_to_uint8(ds.pixel_array, slope, intercept)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img8, mode="L").save(out_path)


def find_by_sop(series_dir: Path, sop_uid: str):
    for f in series_dir.glob("*.dcm"):
        try:
            hdr = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        if str(hdr.SOPInstanceUID) == str(sop_uid):
            return pydicom.dcmread(f)
    return None


def middle_slice(series_dir: Path):
    files = sorted(series_dir.glob("*.dcm"))
    return pydicom.dcmread(files[len(files) // 2]) if files else None


def find_existing_png(series_uid: str):
    for label in ("Malignant", "Benign", "Normal"):
        p = OLD_FULL_ROOT / label / f"{series_uid}.png"
        if p.exists():
            return p
    return None


def main():
    final = pd.read_csv(FINAL_POOL, dtype={"series_uid": str})
    labels = pd.read_csv(SERIES_LABELS, dtype={"series_uid": str, "best_slice_sop_uid": str})
    sop_map = dict(zip(labels["series_uid"], labels["best_slice_sop_uid"]))

    print(f"Series in final pool: {len(final)}", flush=True)
    print(final["canonical_label"].value_counts().to_string(), flush=True)

    rows, n_copy, n_extract, n_fail = [], 0, 0, 0
    total = len(final)
    for i, r in enumerate(final.itertuples(), 1):
        series_uid = str(r.series_uid)
        label = r.canonical_label
        series_dir = RAW_ROOT / series_uid
        out_path = OUT_IMG_ROOT / label / f"{series_uid}.png"

        try:
            if out_path.exists():
                pass
            else:
                cached = find_existing_png(series_uid)
                if cached is not None:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cached, out_path)
                    n_copy += 1
                else:
                    sop = sop_map.get(series_uid)
                    ds = find_by_sop(series_dir, sop) if isinstance(sop, str) and sop and sop != "nan" else None
                    if ds is None:
                        ds = middle_slice(series_dir)
                    if ds is None:
                        raise RuntimeError("no dicom slice found")
                    save_slice(ds, out_path)
                    n_extract += 1

            rows.append({
                "path": str(out_path),
                "canonical_label": label,
                "split_group": r.split_group,
                "series_uid": series_uid,
                "source_dataset": "LIDC-IDRI",
                "worst_mean_malignancy": getattr(r, "worst_mean_malignancy", None),
                "relabeled_from_ambiguous": getattr(r, "relabeled_from_ambiguous", None),
                "pathology_confirmed": getattr(r, "pathology_confirmed", None),
            })
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"  [{i}/{total}] FAILED {series_uid}: {e}", flush=True)
            continue

        if i % 100 == 0 or i == total:
            print(f"  [{i}/{total}] disalin={n_copy} diekstrak={n_extract} gagal={n_fail}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_MANIFEST, index=False)

    print("\n=== Pool whole-slice (label final) ===", flush=True)
    print(df["canonical_label"].value_counts().to_string(), flush=True)
    print(f"Total: {len(df)} | pasien unik: {df['split_group'].nunique()}", flush=True)
    print(f"Manifest: {OUT_MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
