"""Repoint the OOD manifest from the superseded whole-slice folder to the
current one, so the old folder can be removed without breaking the input
validation experiment. Matches by filename, because the label subfolder
changed when the labels were corrected. Writes nothing unless every path
resolves.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MANIFEST = Path("D:/skripsi/Kodingan/outputs/manifests/ood_manifest.csv")
OLD_DIR = "lidc-idri-images"
NEW_ROOT = Path("D:/skripsi/Dataset/lidc-idri-images-full")


def main():
    m = pd.read_csv(MANIFEST)
    lookup = {p.name: str(p) for p in NEW_ROOT.rglob("*.png")}

    def repoint(p: str) -> str:
        parts = Path(p).parts
        if OLD_DIR in parts:  # exact folder name, not the -full/-crop variants
            return lookup.get(Path(p).name, p)
        return p

    m["path"] = m["path"].map(repoint)

    still_old = sum(1 for p in m["path"] if OLD_DIR in Path(p).parts)
    missing = sum(1 for p in m["path"] if not Path(p).exists())
    print(f"baris masih menunjuk folder lama : {still_old}")
    print(f"berkas tidak ditemukan           : {missing}")

    if still_old == 0 and missing == 0:
        m.to_csv(MANIFEST, index=False)
        print("OK manifes diarahkan ulang dan disimpan")
    else:
        print("DIBATALKAN -- manifes tidak diubah")


if __name__ == "__main__":
    main()
