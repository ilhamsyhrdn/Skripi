"""Uji fungsional lapisan validasi input: 12 citra CT paru + 8 citra COCO.

Bukan pelatihan ulang, melainkan pengujian gerbang yang sudah terlatih terhadap
unggahan yang mungkin dilakukan pengguna. Citra CT diambil dari held-out test
set supaya gerbang ini diuji pada citra yang tidak pernah dilihatnya.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.ood_predictor import OODDetector

ROOT = Path("D:/skripsi/Kodingan")
COCO = Path("D:/skripsi/Dataset/Dataset COCO/val2017")
OUT_FIG = ROOT / "outputs/figures/final"
OUT_REP = ROOT / "outputs/reports"
N_CT, N_COCO = 12, 8
SEED = 42


def main():
    det = OODDetector(ROOT / "outputs/models/ood_detector.pt")
    if not det.available:
        raise SystemExit("ood_detector.pt tidak ditemukan")

    test = pd.read_csv(ROOT / "outputs/manifests/test_holdout_full.csv")
    rng = np.random.default_rng(SEED)
    # ambil merata dari ketiga kelas supaya gerbang diuji pada seluruh ragam citra CT
    ct = pd.concat([test[test.canonical_label == k].sample(N_CT // 3, random_state=SEED)
                    for k in ["Malignant", "Benign", "Normal"]], ignore_index=True)
    # citra COCO yang pernah masuk manifes gerbang dikecualikan, supaya pengujian
    # ini benar-benar memakai citra yang belum pernah dilihat model penyaring
    dipakai = {Path(x).name for x in pd.read_csv(ROOT / "outputs/manifests/ood_manifest.csv").path}
    coco_files = [f for f in sorted(COCO.glob("*.jpg")) if f.name not in dipakai]
    assert len(coco_files) >= N_COCO, "citra COCO tak terpakai tidak cukup"
    coco = [coco_files[i] for i in rng.choice(len(coco_files), N_COCO, replace=False)]

    rows = []
    for p, lab in [(Path(r.path), f"Citra CT paru-paru ({r.canonical_label})") for r in ct.itertuples()] + \
                  [(p, "Bukan citra CT paru-paru (objek umum)") for p in coco]:
        with Image.open(p) as im:
            res = det.predict(im)
        benar = res["is_lung_ct"] == lab.startswith("Citra CT")
        rows.append({"berkas": p.name, "kategori": lab,
                     "prob_ct": res["prob_lung_ct"],
                     "keputusan": "Diterima" if res["is_lung_ct"] else "Ditolak",
                     "sesuai": benar, "path": str(p)})

    df = pd.DataFrame(rows)
    n_ok = int(df.sesuai.sum())
    print(f"benar {n_ok}/{len(df)}  ({n_ok/len(df)*100:.1f}%)")
    print(df.drop(columns=["path"]).to_string(index=False))

    # Tiap citra disimpan tersendiri karena naskah menyajikan hasil uji ini
    # sebagai tabel satu baris per citra (No, Gambar, Kategori, Output), bukan
    # sebagai satu gambar kisi.
    dir_thumb = OUT_FIG / "validasi_input"
    dir_thumb.mkdir(parents=True, exist_ok=True)
    for f in dir_thumb.glob("vi_*.png"):
        f.unlink()
    thumbs = []
    for k, r in enumerate(rows, start=1):
        with Image.open(r["path"]) as im:
            im = im.convert("RGB")
            im.thumbnail((420, 420))
            nama = f"vi_{k:02d}.png"
            im.save(dir_thumb / nama)
        thumbs.append(nama)
    df["thumbnail"] = thumbs
    df.to_csv(OUT_REP / "uji_validasi_input.csv", index=False)
    print(f"\nTersimpan: uji_validasi_input.csv dan {len(thumbs)} citra di {dir_thumb.name}/")

if __name__ == "__main__":
    main()
