"""Investigasi: apakah crop latar hitam (di luar tubuh pasien) sebelum resize bisa
membantu akurasi? Dijalankan sebagai eksperimen preprocessing, BUKAN diterapkan ke
pipeline training utama, karena hasilnya menunjukkan dampak yang sangat kecil.

Metodologi: deteksi warna background dari median piksel di 4 pojok citra (adaptif,
karena dataset asli punya background hitam murni sementara IQ-OTH/NCCD punya
background abu-abu gelap seragam), lalu crop ke bounding box piksel yang berbeda
signifikan dari warna background tersebut.

Hasil pada 60 sampel training acak (seed=1): rata-rata pengurangan area cuma 2.8%,
median 0% (lebih dari separuh citra tidak berkurang sama sekali), dan tidak ada
satupun sampel yang berkurang >30%. Kesimpulan: sebagian besar citra di dataset ini
sudah di-crop cukup rapat oleh pembuat dataset aslinya, sehingga preprocessing ini
tidak diharapkan memberi kenaikan akurasi yang berarti -- diputuskan untuk TIDAK
diterapkan ke pipeline training.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image


def crop_body_region(image: Image.Image, diff_threshold: int = 8, margin_frac: float = 0.02) -> Image.Image:
    gray = np.array(image.convert("L")).astype(np.int16)
    h, w = gray.shape
    corner_size = max(5, min(h, w) // 20)
    corners = np.concatenate(
        [
            gray[:corner_size, :corner_size].flatten(),
            gray[:corner_size, -corner_size:].flatten(),
            gray[-corner_size:, :corner_size].flatten(),
            gray[-corner_size:, -corner_size:].flatten(),
        ]
    )
    bg_value = int(np.median(corners))

    diff = np.abs(gray - bg_value)
    mask = diff > diff_threshold
    if not mask.any():
        return image

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]

    margin_y = int(h * margin_frac)
    margin_x = int(w * margin_frac)
    top = max(0, top - margin_y)
    bottom = min(h, bottom + margin_y)
    left = max(0, left - margin_x)
    right = min(w, right + margin_x)

    return image.crop((left, top, right, bottom))


def main():
    root = Path(__file__).resolve().parent.parent
    random.seed(1)
    paths = [p for p in (root / "dataset_split" / "train").rglob("*") if p.is_file()]
    sample_paths = random.sample(paths, 60)

    ratios = []
    for p in sample_paths:
        img = Image.open(p).convert("RGB")
        cropped = crop_body_region(img)
        w, h = img.size
        cw, ch = cropped.size
        ratios.append((cw * ch) / (w * h))

    ratios = np.array(ratios)
    print("Area ratio (cropped/original) pada 60 sampel training acak:")
    print(f"  mean={ratios.mean():.4f} median={np.median(ratios):.4f} min={ratios.min():.4f}")
    print(f"  fraksi dgn pengurangan area >10%: {(ratios < 0.9).mean():.4f}")
    print(f"  fraksi dgn pengurangan area >30%: {(ratios < 0.7).mean():.4f}")
    print("\nKesimpulan: dampak crop terlalu kecil untuk diharapkan menaikkan akurasi -- tidak diterapkan.")


if __name__ == "__main__":
    main()
