"""Bandingkan ketiga konfigurasi data augmentation.

Ketiga kelompok model berbagi seluruh pipeline yang sama kecuali pada bagian
transformasi augmentasinya, sehingga selisih di antara ketiganya adalah
kontribusi augmentasi itu sendiri. Tiga hal diukur:

1. Performa pada held-out test set lewat alur ensemble stacking yang sama.
2. Jurang overfitting, yaitu seberapa jauh akurasi data latih meninggalkan
   akurasi validasi pada epoch terbaik tiap model.
3. Kebermaknaan statistik selisih antar konfigurasi lewat uji McNemar, karena
   ketiganya diuji pada citra uji yang sama persis sehingga prediksinya
   berpasangan.

Dijalankan setelah train_cv.py selesai untuk ketiga konfigurasi, dan setelah
evaluate_models.py serta stacking_ensemble.py dijalankan untuk masing-masing.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

REPORTS = Path("D:/skripsi/Kodingan/outputs/reports")
MANIFESTS = Path("D:/skripsi/Kodingan/outputs/manifests")
CLASS_NAMES = ["Benign", "Malignant", "Normal"]
MAL = CLASS_NAMES.index("Malignant")
VARIANT = "full"
THRESHOLD = 0.50

# kunci pendek -> (nama tampil, akhiran berkas hasil). Kunci pendek dipakai
# sebagai indeks CSV karena make_all_figures memanggilnya lewat nama itu.
KONFIGURASI = {
    "tanpa": ("Tanpa augmentasi", f"{VARIANT}_noaug"),
    "ringan_ct": ("Augmentasi ringan CT", f"{VARIANT}_augct"),
    "penuh": ("Augmentasi penuh", VARIANT),
}


def _y_true() -> np.ndarray:
    test = pd.read_csv(MANIFESTS / f"test_holdout_{VARIANT}.csv")
    return test["canonical_label"].map({c: i for i, c in enumerate(CLASS_NAMES)}).values


def prediksi(suffix: str) -> tuple[np.ndarray, np.ndarray]:
    """Probabilitas stacking dan prediksinya, memakai aturan ambang yang sama
    persis dengan stacking_ensemble.py."""
    probs = np.load(REPORTS / f"stacking_probs_{suffix}.npy")
    pred = np.where(probs[:, MAL] >= THRESHOLD, MAL, probs.argmax(1))
    return probs, pred


def metrik(suffix: str, y: np.ndarray) -> dict:
    probs, pred = prediksi(suffix)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "cancer_recall": recall_score(y, pred, labels=[MAL], average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, probs, multi_class="ovr", average="macro"),
    }


def jurang_overfitting(suffix: str) -> dict:
    """Rata-rata selisih akurasi latih dikurangi validasi pada epoch terbaik.

    Pola nama disebut per arsitektur, bukan lewat awalan saja, karena
    `history_full_*` juga cocok dengan `history_full_noaug_*` dan
    `history_full_augct_*` sehingga ketiga konfigurasi akan tercampur.
    """
    berkas = sorted(f for arch in ("efficientnet_b0", "resnet50")
                    for f in REPORTS.glob(f"history_{suffix}_{arch}_fold*.json"))
    assert len(berkas) == 10, f"{suffix}: ditemukan {len(berkas)} riwayat, seharusnya 10"
    latih, validasi = [], []
    for f in berkas:
        h = pd.DataFrame(json.load(open(f)))
        best = h.loc[h["macro_f1"].idxmax()]
        latih.append(best["train_acc"])
        validasi.append(best["acc"])
    latih, validasi = np.array(latih), np.array(validasi)
    return {"n_model": len(berkas), "train_acc": float(latih.mean()),
            "val_acc": float(validasi.mean()), "gap": float((latih - validasi).mean())}


def mcnemar(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> dict:
    """Uji McNemar eksak: hanya citra yang salah satu benar dan lainnya salah
    yang membawa informasi; sisanya tidak membedakan kedua model."""
    a_benar, b_benar = pred_a == y, pred_b == y
    n01 = int((~a_benar & b_benar).sum())   # hanya B benar
    n10 = int((a_benar & ~b_benar).sum())   # hanya A benar
    p = binomtest(n10, n10 + n01, 0.5).pvalue if (n10 + n01) else 1.0
    return {"hanya_A_benar": n10, "hanya_B_benar": n01, "n_berbeda": n10 + n01, "p_value": p}


def main():
    y = _y_true()

    baris = []
    for kunci, (nama, suffix) in KONFIGURASI.items():
        m = metrik(suffix, y)
        g = jurang_overfitting(suffix)
        baris.append({"konfigurasi": kunci, "nama": nama, **m,
                      "train_acc": g["train_acc"], "val_acc": g["val_acc"],
                      "jurang_overfitting": g["gap"]})

    df = pd.DataFrame(baris)
    print("=== Performa data uji (ensemble stacking, ambang 0,50) dan jurang overfitting ===")
    tampil = df.copy()
    for c in ["accuracy", "cancer_recall", "train_acc", "val_acc", "jurang_overfitting"]:
        tampil[c] = (tampil[c] * 100).round(2)
    tampil["macro_f1"] = tampil["macro_f1"].round(4)
    tampil["roc_auc"] = tampil["roc_auc"].round(4)
    print(tampil.to_string(index=False), flush=True)

    print("\n=== Uji McNemar antar konfigurasi (442 citra uji yang sama) ===")
    uji = []
    for a, b in combinations(KONFIGURASI, 2):
        r = mcnemar(prediksi(KONFIGURASI[a][1])[1], prediksi(KONFIGURASI[b][1])[1], y)
        uji.append({"konfigurasi_A": KONFIGURASI[a][0], "konfigurasi_B": KONFIGURASI[b][0], **r,
                    "kesimpulan": "berbeda bermakna" if r["p_value"] < 0.05 else "tidak berbeda bermakna"})
        print(f"{KONFIGURASI[a][0]} vs {KONFIGURASI[b][0]}: hanya A benar={r['hanya_A_benar']}, hanya B benar={r['hanya_B_benar']}, "
              f"p={r['p_value']:.4f} -> {uji[-1]['kesimpulan']}", flush=True)

    df.to_csv(REPORTS / f"ablasi_augmentasi_{VARIANT}.csv", index=False)
    pd.DataFrame(uji).to_csv(REPORTS / f"ablasi_augmentasi_mcnemar_{VARIANT}.csv", index=False)
    print(f"\nTersimpan: ablasi_augmentasi_{VARIANT}.csv dan ablasi_augmentasi_mcnemar_{VARIANT}.csv")


if __name__ == "__main__":
    main()
