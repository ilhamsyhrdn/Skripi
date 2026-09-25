"""Extract one representative 2D PNG slice per LIDC-IDRI series and build a
manifest CSV compatible with the training pipeline (path, canonical_label,
split_group). Resumable: series whose output PNG already exists are skipped.

- Malignant / Benign series: the DICOM slice matching the worst consensus
  nodule's best_slice_sop_uid (from lidc_series_labels.csv).
- Normal series (zero characterized nodules): the middle slice of the series.

Lung window (level -600, width 1500) converts raw Hounsfield-unit pixel data
to a viewable 8-bit grayscale image.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

RAW_ROOT = Path("D:/skripsi/Dataset/lidc-idri")
LABELS_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/lidc_series_labels.csv")
OUT_IMG_ROOT = Path("D:/skripsi/Dataset/lidc-idri-images")
OUT_MANIFEST = Path("D:/skripsi/Kodingan/outputs/manifests/lidc_canonical_pool.csv")

WINDOW_LEVEL = -600
WINDOW_WIDTH = 1500


def hu_to_uint8(pixel_array: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    hu = pixel_array.astype(np.float32) * slope + intercept
    lo = WINDOW_LEVEL - WINDOW_WIDTH / 2
    hi = WINDOW_LEVEL + WINDOW_WIDTH / 2
    hu = np.clip(hu, lo, hi)
    img = (hu - lo) / (hi - lo) * 255.0
    return img.astype(np.uint8)


def save_slice(ds, out_path: Path) -> None:
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
        if hdr.SOPInstanceUID == sop_uid:
            return pydicom.dcmread(f)
    return None


def middle_slice(series_dir: Path):
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        return None
    mid = files[len(files) // 2]
    return pydicom.dcmread(mid)


def existing_png(series_uid: str) -> Path | None:
    for label in ("Malignant", "Benign", "Normal"):
        p = OUT_IMG_ROOT / label / f"{series_uid}.png"
        if p.exists():
            return p
    return None


def patient_id_from_png_or_dicom(series_uid: str, series_dir: Path, png_path: Path | None) -> str:
    if png_path is not None:
        # re-read one dicom header just for patient id (cheap, header-only)
        f = next(series_dir.glob("*.dcm"), None)
        if f is not None:
            hdr = pydicom.dcmread(f, stop_before_pixels=True)
            return str(hdr.PatientID)
    return ""


def main():
    full_labels_df = pd.read_csv(LABELS_CSV, dtype={"series_uid": str, "best_slice_sop_uid": str})
    excluded_uids = set(full_labels_df.loc[full_labels_df["canonical_label"] == "Excluded", "series_uid"])
    labels_df = full_labels_df[full_labels_df["canonical_label"].isin(["Malignant", "Benign"])]
    labeled_map = {r.series_uid: r for r in labels_df.itertuples()}

    all_series = sorted(
        p.name for p in RAW_ROOT.iterdir() if p.is_dir() and p.name not in excluded_uids
    )
    total = len(all_series)
    print(f"Total series to process: {total}", flush=True)

    rows = []
    n_ok = n_skip = n_fail = 0
    for i, series_uid in enumerate(all_series, 1):
        series_dir = RAW_ROOT / series_uid
        rec = labeled_map.get(series_uid)
        label = rec.canonical_label if rec is not None else "Normal"

        out_path = OUT_IMG_ROOT / label / f"{series_uid}.png"

        try:
            if out_path.exists():
                # already extracted in a previous run -- just need patient id for manifest
                f = next(series_dir.glob("*.dcm"), None)
                hdr = pydicom.dcmread(f, stop_before_pixels=True)
                patient_id = str(hdr.PatientID)
                n_skip += 1
            else:
                if rec is not None:
                    ds = find_by_sop(series_dir, rec.best_slice_sop_uid)
                    if ds is None:
                        ds = middle_slice(series_dir)
                else:
                    ds = middle_slice(series_dir)
                if ds is None:
                    raise RuntimeError("no dicom slice found")
                patient_id = str(ds.PatientID)
                save_slice(ds, out_path)
                n_ok += 1

            rows.append({
                "path": str(out_path),
                "canonical_label": label,
                "split_group": patient_id,
                "series_uid": series_uid,
                "source_dataset": "LIDC-IDRI",
                "worst_mean_malignancy": getattr(rec, "worst_mean_malignancy", None),
            })
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"  [{i}/{total}] FAILED {series_uid}: {e}", flush=True)
            continue

        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] new_ok={n_ok} skipped_existing={n_skip} fail={n_fail}", flush=True)

    df = pd.DataFrame(rows)
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_MANIFEST, index=False)

    print("\n=== Final image counts ===", flush=True)
    print(df["canonical_label"].value_counts(), flush=True)
    n_patients = df["split_group"].nunique()
    print(f"Unique patients (split_group): {n_patients}", flush=True)
    print(f"Saved manifest: {OUT_MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
