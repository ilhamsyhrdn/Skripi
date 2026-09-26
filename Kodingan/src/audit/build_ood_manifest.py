"""Bangun manifes OOD (out-of-distribution) untuk lapisan validasi input.

Kelas positif "Lung_CT" diambil dari citra CT LIDC-IDRI irisan utuh, dan kelas
negatif "Bukan_Lung_CT" dari sampel acak COCO val2017 dengan jumlah yang sama.

Pembagian train/val/test mengikuti pembagian eksperimen utama: citra LIDC yang
masuk held-out test set pada manifes utama juga menjadi test set di sini,
sehingga gerbang validasi input tidak pernah diuji memakai citra yang dipakai
melatihnya. Sisanya dibagi 90/10 menjadi train dan val.

Sumber positif sengaja dibaca dari manifes utama yang sudah melewati penyaringan
tag Modality. Tanpa itu, seri rontgen dada (DX/CR) milik LIDC-IDRI ikut terbawa
sebagai contoh "citra CT paru", padahal windowing HU menjenuhkannya menjadi
putih polos, sehingga gerbang justru dilatih menerima citra kosong.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

MANIFESTS = Path("D:/skripsi/Kodingan/outputs/manifests")
TEST_CSV = MANIFESTS / "test_holdout_full.csv"
TRAINVAL_CSV = MANIFESTS / "trainval_folds_full.csv"
COCO_DIR = Path("D:/skripsi/Dataset/Dataset COCO/val2017")
OUT_CSV = MANIFESTS / "ood_manifest.csv"
SEED = 42
VAL_FRACTION = 0.10


def _lidc(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    return df[df["source_dataset"] == "LIDC-IDRI"].reset_index(drop=True)


def main():
    test_df = _lidc(TEST_CSV)
    trainval_df = _lidc(TRAINVAL_CSV)
    assert len(test_df) and len(trainval_df), "manifes utama kosong; jalankan build_full_dataset.py dulu"

    n_val = max(1, round(len(trainval_df) * VAL_FRACTION))
    shuffled = trainval_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_part, train_part = shuffled.iloc[:n_val], shuffled.iloc[n_val:]

    pos = [{"path": r["path"], "label": 1, "class_name": "Lung_CT", "split": s}
           for part, s in ((test_df, "test"), (train_part, "train"), (val_part, "val"))
           for _, r in part.iterrows()]

    coco_files = sorted(COCO_DIR.glob("*.jpg"))
    if len(coco_files) < len(pos):
        raise RuntimeError(f"Citra COCO kurang: ada {len(coco_files)}, butuh {len(pos)}")
    rng = random.Random(SEED)
    sampled = coco_files.copy()
    rng.shuffle(sampled)
    sampled = sampled[:len(pos)]

    # jumlah tiap split pada kelas negatif dibuat persis sama dengan kelas positif
    urutan = ["test"] * len(test_df) + ["val"] * n_val + ["train"] * (len(pos) - len(test_df) - n_val)
    neg = [{"path": str(p), "label": 0, "class_name": "Bukan_Lung_CT", "split": s}
           for p, s in zip(sampled, urutan)]

    df = pd.concat([pd.DataFrame(pos), pd.DataFrame(neg)], ignore_index=True)

    tumpang_tindih = set(test_df["path"]) & set(pd.concat([train_part, val_part])["path"])
    assert not tumpang_tindih, "citra uji bocor ke data latih gerbang validasi"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(df.groupby(["split", "class_name"]).size().to_string(), flush=True)
    print(f"Total: {len(df)} citra ({len(pos)} positif + {len(neg)} negatif)", flush=True)
    print(f"Tersimpan: {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
