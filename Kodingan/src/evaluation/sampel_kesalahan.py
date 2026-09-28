"""Siapkan citra sampel kesalahan prediksi untuk tabel analisis di Bab IV.

Dari tiap pola kesalahan (misalnya Benign yang diprediksi Malignant) diambil
satu citra, yaitu kesalahan yang diprediksi dengan keyakinan paling tinggi.
Kesalahan yang diyakini penuh oleh model paling berguna untuk dianalisis,
karena menunjukkan pola visual yang benar-benar menyesatkan model, bukan
sekadar kasus yang memang berada di tepi ambang.

Setiap citra diberi pita keterangan "Asli" dan "Prediksi" di bagian atasnya,
lalu dicatat bersama sumber data dan probabilitas ketiga kelas agar analisis
di naskah dapat ditulis berdasarkan angka, bukan dugaan.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("D:/skripsi/Kodingan")
KELAS = ["Benign", "Malignant", "Normal"]
MAL = KELAS.index("Malignant")
AMBANG = 0.50
SUFFIX = "full_augct"
OUT_DIR = ROOT / "outputs/figures/final/sampel_kesalahan"
OUT_CSV = ROOT / "outputs/reports/sampel_kesalahan.csv"


def pita(img: Image.Image, asli: str, prediksi: str) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((440, 440))
    # tinggi huruf 6,8% lebar citra: pada lebar cetak 4 cm di naskah setara
    # sekitar 7,6 pt, sehingga keterangan tetap terbaca ketika dicetak
    ukuran = max(12, round(img.width * 0.068))
    jarak = round(ukuran * 1.2)
    tinggi_pita = 2 * jarak + 8
    kanvas = Image.new("RGB", (img.width, img.height + tinggi_pita), "white")
    kanvas.paste(img, (0, tinggi_pita))
    d = ImageDraw.Draw(kanvas)
    f = ImageFont.truetype("arialbd.ttf", ukuran)
    for k, teks in enumerate((f"Asli: {asli}", f"Prediksi: {prediksi}")):
        w = d.textlength(teks, font=f)
        assert w <= img.width - 8, f"keterangan terlalu lebar: {teks}"
        d.text(((img.width - w) / 2, 4 + k * jarak), teks, fill="black", font=f)
    return kanvas


def main():
    test = pd.read_csv(ROOT / "outputs/manifests/test_holdout_full.csv")
    P = np.load(ROOT / f"outputs/reports/stacking_probs_{SUFFIX}.npy")
    y = test["canonical_label"].map({c: i for i, c in enumerate(KELAS)}).values
    pred = np.where(P[:, MAL] >= AMBANG, MAL, P.argmax(1))

    test = test.assign(asli=[KELAS[i] for i in y], prediksi=[KELAS[i] for i in pred],
                       keyakinan=P[np.arange(len(P)), pred],
                       p_benign=P[:, 0], p_malignant=P[:, 1], p_normal=P[:, 2])
    salah = test[pred != y].copy()
    salah["pola"] = salah.asli + " -> " + salah.prediksi

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUT_DIR.glob("sk_*.png"):
        f.unlink()

    # urutan pola mengikuti jumlah kesalahannya, dari yang paling sering
    urutan = salah.pola.value_counts().index
    baris = []
    for k, pola in enumerate(urutan, start=1):
        g = salah[salah.pola == pola].sort_values("keyakinan", ascending=False)
        r = g.iloc[0]
        with Image.open(r.path) as im:
            nama = f"sk_{k:02d}.png"
            pita(im, r.asli, r.prediksi).save(OUT_DIR / nama)
        sumber = "LIDC-IDRI" if "LIDC" in r.source_dataset else "Kaggle"
        baris.append({"no": k, "pola": pola, "jumlah_pola": len(g), "berkas": Path(r.path).name,
                      "sumber": sumber, "asli": r.asli, "prediksi": r.prediksi,
                      "keyakinan": round(r.keyakinan, 4), "p_benign": round(r.p_benign, 4),
                      "p_malignant": round(r.p_malignant, 4), "p_normal": round(r.p_normal, 4),
                      "thumbnail": nama, "path": r.path})

    df = pd.DataFrame(baris)
    df.to_csv(OUT_CSV, index=False)
    print(df.drop(columns=["path", "berkas"]).to_string(index=False))
    print(f"\nTersimpan: {OUT_CSV.name} dan {len(df)} citra di {OUT_DIR.name}/")


if __name__ == "__main__":
    main()
