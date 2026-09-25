"""v2 extraction: crop a fixed-size window AROUND THE NODULE (instead of
saving the whole 512x512 slice) so the pathology is not lost in a tiny
fraction of a downscaled 224x224 image.

Crop window is the SAME PHYSICAL SIZE for every class (including Normal,
centered on the image/lung field instead of a nodule) -- this is important:
if only Malignant/Benign were cropped tight and Normal stayed as a wide
full-frame shot, the model could learn to classify by "zoom level" alone
(a shortcut/leakage artifact) instead of by actual pathology. Keeping the
crop size and framing method consistent across classes avoids that shortcut.

Output goes to a SEPARATE folder (lidc-idri-images-crop) and manifest
(lidc_canonical_pool_crop.csv) so the original whole-slice extraction is not
destroyed -- the two are compared before deciding which one the thesis uses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

RAW_ROOT = Path("D:/skripsi/Dataset/lidc-idri")
LABELS_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/lidc_series_labels.csv")
OUT_IMG_ROOT = Path("D:/skripsi/Dataset/lidc-idri-images-crop")
OUT_MANIFEST = Path("D:/skripsi/Kodingan/outputs/manifests/lidc_canonical_pool_crop.csv")

WINDOW_LEVEL = -600
WINDOW_WIDTH = 1500
CROP_PX = 224  # saved directly at network input resolution, same for every class


def hu_to_uint8(pixel_array: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    hu = pixel_array.astype(np.float32) * slope + intercept
    lo = WINDOW_LEVEL - WINDOW_WIDTH / 2
    hi = WINDOW_LEVEL + WINDOW_WIDTH / 2
    hu = np.clip(hu, lo, hi)
    img = (hu - lo) / (hi - lo) * 255.0
    return img.astype(np.uint8)


def crop_around(img8: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h, w = img8.shape
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x0 = max(0, min(x0, w - size))
    y0 = max(0, min(y0, h - size))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x0 + size), min(h, y0 + size)
    return img8[y0:y1, x0:x1]


def save_crop(ds, cx: float, cy: float, out_path: Path) -> None:
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    img8 = hu_to_uint8(ds.pixel_array, slope, intercept)
    crop = crop_around(img8, cx, cy, CROP_PX)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop, mode="L").save(out_path)


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


def main():
    full_labels_df = pd.read_csv(LABELS_CSV, dtype={"series_uid": str, "best_slice_sop_uid": str})
    # keep ALL characterized-nodule series, including "Excluded" (ambiguous
    # score==3) -- those get a temporary "Excluded_TBD" label here and are
    # relabeled by src/audit/relabel_ambiguous.py (nearest-neighbour transfer
    # from Zhang et al. 2022, MICCAI Workshop) instead of being thrown away.
    labeled_map = {r.series_uid: r for r in full_labels_df.itertuples()}

    all_series = sorted(p.name for p in RAW_ROOT.iterdir() if p.is_dir())
    total = len(all_series)
    print(f"Total series to process: {total}", flush=True)

    rows = []
    n_ok = n_skip = n_fail = 0
    for i, series_uid in enumerate(all_series, 1):
        series_dir = RAW_ROOT / series_uid
        rec = labeled_map.get(series_uid)
        if rec is None:
            label = "Normal"
        elif rec.canonical_label == "Excluded":
            label = "Excluded_TBD"
        else:
            label = rec.canonical_label
        out_path = OUT_IMG_ROOT / label / f"{series_uid}.png"

        try:
            if out_path.exists():
                f = next(series_dir.glob("*.dcm"), None)
                hdr = pydicom.dcmread(f, stop_before_pixels=True)
                patient_id = str(hdr.PatientID)
                n_skip += 1
            else:
                if rec is not None:
                    ds = find_by_sop(series_dir, rec.best_slice_sop_uid)
                    if ds is None:
                        ds = middle_slice(series_dir)
                    cx, cy = rec.nodule_cx, rec.nodule_cy
                else:
                    ds = middle_slice(series_dir)
                    if ds is None:
                        raise RuntimeError("no dicom slice found")
                    h, w = ds.pixel_array.shape
                    cx, cy = w / 2, h / 2
                if ds is None:
                    raise RuntimeError("no dicom slice found")
                patient_id = str(ds.PatientID)
                save_crop(ds, cx, cy, out_path)
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

        if i % 100 == 0 or i == total:
            print(f"  [{i}/{total}] new_ok={n_ok} skipped_existing={n_skip} fail={n_fail}", flush=True)

    df = pd.DataFrame(rows)
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_MANIFEST, index=False)
    print("\n=== Final image counts (cropped) ===", flush=True)
    print(df["canonical_label"].value_counts(), flush=True)
    print(f"Saved manifest: {OUT_MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
