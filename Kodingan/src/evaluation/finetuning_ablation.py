"""Ukur kontribusi fine-tuning per model: fase A dibandingkan fase B.

Fase A melatih kepala klasifikasi saja dengan backbone dibekukan, sedangkan
fase B membuka tiga blok teratas backbone dengan laju pembelajaran jauh lebih
kecil. Keduanya berjalan berurutan pada model yang sama, sehingga selisih
macro-F1 validasi terbaik di antara kedua fase adalah kontribusi fine-tuning
pada model tersebut.

Angka diambil dari riwayat pelatihan yang ditulis train_cv.py, bukan dihitung
ulang, supaya identik dengan nilai yang dipakai saat memilih checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPORTS = Path("D:/skripsi/Kodingan/outputs/reports")
ARCHS = ("efficientnet_b0", "resnet50")
FOLDS = range(5)


def terbaik(riwayat: list[dict], fase: str) -> dict | None:
    """Epoch dengan macro-F1 validasi tertinggi pada satu fase."""
    epochs = [e for e in riwayat if e["phase"] == fase]
    return max(epochs, key=lambda e: e["macro_f1"]) if epochs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="full",
                   help="akhiran berkas riwayat, misal full, full_noaug, full_augct")
    args = p.parse_args()

    baris = []
    for arch in ARCHS:
        for fold in FOLDS:
            f = REPORTS / f"history_{args.suffix}_{arch}_fold{fold}.json"
            assert f.exists(), f"riwayat tidak ditemukan: {f.name}"
            h = json.load(open(f))
            a, b = terbaik(h, "A-head"), terbaik(h, "B-finetune")
            assert a and b, f"{f.name}: salah satu fase kosong"
            baris.append({
                "model": f"{arch}_fold{fold}",
                "f1_fase_A": a["macro_f1"], "f1_fase_B": b["macro_f1"],
                "acc_fase_A": a["acc"], "acc_fase_B": b["acc"],
                "rec_fase_A": a["cancer_recall"], "rec_fase_B": b["cancer_recall"],
                "delta_f1": b["macro_f1"] - a["macro_f1"],
            })

    df = pd.DataFrame(baris)
    out = REPORTS / f"ablasi_finetuning_{args.suffix}.csv"
    df.to_csv(out, index=False)

    naik = int((df["delta_f1"] > 0).sum())
    print(df[["model", "f1_fase_A", "f1_fase_B", "delta_f1"]].round(4).to_string(index=False), flush=True)
    print(f"\nNaik pada {naik} dari {len(df)} model | "
          f"rata-rata selisih macro-F1 = {df['delta_f1'].mean():+.4f}", flush=True)
    print(f"Tersimpan: {out.name}", flush=True)


if __name__ == "__main__":
    main()
