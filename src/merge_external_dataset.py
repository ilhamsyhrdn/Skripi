"""Menggabungkan dataset eksternal IQ-OTH/NCCD ke dalam training set.

CATATAN PENTING (lihat commit history untuk detail lengkap): percobaan pertama
membagi IQ-OTH/NCCD 80/20 (80% training, 20% "external holdout") gagal karena
dataset ini terdiri dari beberapa slice CT per pasien dengan penomoran file
berurutan (contoh: "Malignant case (93).jpg" dan "Malignant case (94).jpg").
Slice-slice yang bersebelahan itu sangat mirip secara visual (cosine similarity
>0.99, bahkan 1.0000 pada resolusi rendah) karena berasal dari pasien yang sama.
Split acak di level gambar menyebabkan slice dari pasien yang sama tersebar ke
train dan "holdout", sehingga evaluasi "holdout" itu bocor (akurasi 98.6% palsu)
dan tidak valid sebagai ukuran generalisasi.

Karena dataset ini tidak menyertakan ID pasien/case yang bisa dipakai untuk
group-aware split, tidak ada cara aman untuk menyisakan sebagian sebagai
external holdout dari dataset ini. Solusinya: seluruh citra unik (setelah
dedup + drop conflicting label) digabungkan ke training set, dan generalisasi
ke luar dataset asli tetap diukur lewat validasi eksternal zero-shot yang
SUDAH dilakukan sebelum data ini pernah disentuh sama sekali oleh training
(lihat reports/external_validation_iqothnccd.json, dihasilkan oleh
src/eval_external.py dengan model v5 yang belum pernah melihat citra ini).
"""

import hashlib
import shutil
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))


def main():
    import numpy as np
    import pandas as pd
    from PIL import Image

    ROOT = SRC_DIR.parent
    EXTERNAL_ROOT = (
        ROOT
        / "dataset_external"
        / "iqothnccd"
        / "The IQ-OTHNCCD lung cancer dataset"
        / "The IQ-OTHNCCD lung cancer dataset"
    )
    DATASET_SPLIT = ROOT / "dataset_split"
    REPORTS_DIR = ROOT / "reports"

    folder_to_label = {
        "Malignant cases": "cancer",
        "Bengin cases": "no_cancer",
        "Normal cases": "no_cancer",
    }

    records = []
    for folder, label in folder_to_label.items():
        for p in sorted((EXTERNAL_ROOT / folder).glob("*.jpg")):
            with Image.open(p) as im:
                arr = np.array(im.convert("RGB").resize((64, 64)))
            h = hashlib.md5(arr.tobytes()).hexdigest()
            records.append({"path": str(p), "filename": p.name, "label": label, "phash": h})

    manifest = pd.DataFrame(records)
    print(f"IQ-OTH/NCCD total citra: {len(manifest)}")

    groups = manifest.groupby("phash")
    conflicting = {h for h, g in groups if g["label"].nunique() > 1}
    print(f"Grup label bertentangan: {len(conflicting)}")

    clean = manifest[~manifest["phash"].isin(conflicting)].copy()
    deduped = clean.drop_duplicates(subset="phash", keep="first").reset_index(drop=True)
    print(f"Setelah dedup: {len(deduped)} citra unik")
    print(deduped["label"].value_counts())

    for label in ["cancer", "no_cancer"]:
        (DATASET_SPLIT / "train" / label).mkdir(parents=True, exist_ok=True)

    for row in deduped.itertuples(index=False):
        dest = DATASET_SPLIT / "train" / row.label / f"iqothnccd__{row.filename}"
        if not dest.exists():
            shutil.copy2(row.path, dest)

    for label in ["cancer", "no_cancer"]:
        n = len(list((DATASET_SPLIT / "train" / label).glob("*")))
        print(f"train/{label}: {n}")

    deduped.to_csv(REPORTS_DIR / "iqothnccd_merged_into_train_manifest.csv", index=False)
    print("\nManifest disimpan ke reports/iqothnccd_merged_into_train_manifest.csv")


if __name__ == "__main__":
    main()
