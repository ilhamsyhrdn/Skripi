"""Catat jenis pemindaian (Modality) tiap seri LIDC-IDRI.

LIDC-IDRI menyimpan CT dan foto rontgen dada dalam format DICOM yang sama dan
struktur folder yang sama; yang membedakan hanya tag Modality di dalam berkas.
Tanpa pemeriksaan ini, seri rontgen (DX/CR) ikut terbawa ke pool dan diproses
seolah citra CT, padahal skala nilai pikselnya berbeda sehingga windowing HU
menjenuhkannya menjadi putih polos.

Keluaran dipakai build_full_dataset.py untuk menyaring pool agar benar-benar
hanya berisi citra Computed Tomography sebagaimana dinyatakan judul penelitian.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pydicom

RAW_ROOT = Path("D:/skripsi/Dataset/lidc-idri")
OUT_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/lidc_series_modality.csv")


def main():
    folders = sorted(p for p in RAW_ROOT.iterdir() if p.is_dir())
    print(f"Memindai {len(folders)} folder seri...", flush=True)

    rows = []
    for i, p in enumerate(folders, 1):
        files = list(p.glob("*.dcm"))
        if not files:
            continue
        try:
            ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        except Exception as e:
            print(f"  [lewat] {p.name[:24]}: {e}", flush=True)
            continue
        rows.append({
            "uid": p.name,
            "pid": str(getattr(ds, "PatientID", "")),
            "modality": str(getattr(ds, "Modality", "?")),
            "n": len(files),
            "deskripsi": str(getattr(ds, "SeriesDescription", "")),
        })
        if i % 200 == 0:
            print(f"  {i}/{len(folders)}", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n=== seri menurut modality ===", flush=True)
    print(df.groupby("modality").agg(seri=("uid", "size"), median_irisan=("n", "median")).to_string(), flush=True)
    bukan_ct = df[df.modality != "CT"]
    print(f"\nBukan CT: {len(bukan_ct)} seri ({len(bukan_ct)/len(df)*100:.1f}%) "
          f"dari {df.pid.nunique()} pasien", flush=True)
    print(f"Tersimpan: {OUT_CSV.name}", flush=True)


if __name__ == "__main__":
    main()
