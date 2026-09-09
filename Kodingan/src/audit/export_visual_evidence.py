"""
Copies real example image files into `Kejanggalan Dataset/Bukti Visual/<kategori>/` so every
claim in the Kejanggalan Dataset markdown docs has actual, openable image evidence next to it
-- not just numbers in a CSV.

Run:  Kodingan/.venv/Scripts/python.exe -m src.audit.export_visual_evidence
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = REPO_ROOT / "Kodingan" / "outputs" / "manifests"
OUT_ROOT = REPO_ROOT / "Kejanggalan Dataset" / "Bukti Visual"


def safe_name(source_dataset: str, filename: str) -> str:
    short = source_dataset.split("(")[0].strip().replace(" ", "_")[:40]
    return f"{short}__{filename}"


def copy_files(rows: pd.DataFrame, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for _, r in rows.iterrows():
        target = dest / safe_name(r["source_dataset"], r["filename"])
        if not target.exists():
            shutil.copy2(r["path"], target)


def main() -> None:
    df = pd.read_csv(MANIFEST_DIR / "full_manifest.csv")
    dup = pd.read_csv(MANIFEST_DIR / "duplicate_groups.csv")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1) Leakage / duplikat lintas dataset -- satu citra yang sama muncul
    #    dengan nama berbeda di banyak dataset "berbeda"
    # ------------------------------------------------------------------ #
    d1 = OUT_ROOT / "01_Leakage_Duplikat_Lintas_Dataset"
    grp53 = dup[dup["group_id"] == 53].drop_duplicates("source_dataset")
    copy_files(grp53, d1 / "contoh_01_normal_case_1_muncul_di_6_dataset")
    grp108 = dup[dup["group_id"] == 108].drop_duplicates("source_dataset")
    copy_files(grp108, d1 / "contoh_02_adenocarcinoma_muncul_di_4_dataset")
    (d1 / "README.md").write_text(
        """# Leakage / Duplikat Lintas Dataset

Setiap subfolder berisi **citra CT yang secara pixel identik atau sama persis**, tetapi
disimpan dengan nama file berbeda di dataset Kaggle yang berbeda-beda -- seolah-olah data baru,
padahal isinya sama.

## contoh_01_normal_case_1_muncul_di_6_dataset/
Satu citra "Normal case (1)" dari studi IQ-OTH/NCCD asli, ditemukan **identik** (MD5/perceptual
hash sama) di **6 dari 7 dataset** yang diunduh:
- `The_IQ_OTHNCCD_lung_cancer_dataset____Hamdalla_F.__Al_Yasriy_)__Normal case (1).jpg` (sumber asli)
- `IQ_OTHNCCD___Lung_Cancer_Dataset__Aditya_Mahimkar___Normal case (1).jpg`
- `CT_Scan_Images_of_Lung_Cancer_Patients._MD._NAFEES_IMTIAZ___Normal cases (1).jpg`
- `Lung_cancer_dataset__IQ_OTHNCCD__Waseim_Nagah_Hennes___003828_02_01_174.png`
- `Chest_CT_Scan_images_Dataset__Mohamed_Hany___10 (2) - Copy.png`
- `CT_Scan_Images_for_Lung_Cancer__Dishan_rathi20___10 (2) - Copy.png`

Total anggota grup ini di seluruh dataset: **2.017 salinan** dari 1 citra yang sama (lihat
`Kodingan/outputs/manifests/duplicate_groups.csv`, `group_id=53`).

## contoh_02_adenocarcinoma_muncul_di_4_dataset/
Satu citra adenocarcinoma yang sama muncul di 4 dataset berbeda dengan 4 skema penamaan berbeda
(`000050 (3).png`, `adenocarcinoma (178).png`, `0000051.png`, dst.) -- bukti bahwa dataset
"Mohamed Hany", "Dishan Rathi", "Nafees Imtiaz", dan "Hennes" bersumber dari data yang sama.

**Kalau dibiarkan dan digabung lalu displit acak per gambar**, salinan-salinan ini bisa saja
terpisah -- satu di train, satu di test -- membuat model seolah-olah "mengenali" gambar test
padahal sudah pernah melihat salinan pixel-identiknya saat training. Lihat
`Kejanggalan Dataset/01_duplikasi_lintas_dataset.md` untuk penjelasan lengkap.
""",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # 2) Duplikat copy-paste Windows di DALAM satu dataset
    # ------------------------------------------------------------------ #
    d2 = OUT_ROOT / "02_Duplikat_Copy_Paste_Windows"
    hany_10 = df[
        (df["source_dataset"].str.contains("Mohamed Hany"))
        & (df["raw_folder"] == "test/normal")
        & (df["case_key"].str.endswith("::10"))
    ]
    copy_files(hany_10, d2 / "contoh_01_folder_normal_kasus_10")
    (d2 / "README.md").write_text(
        """# Duplikat Copy-Paste Windows (di dalam 1 dataset, 1 folder kelas)

Folder `contoh_01_folder_normal_kasus_10/` berisi ke-8 file yang ditemukan di
`Chest CT-Scan images Dataset (Mohamed Hany)/test/normal/` dengan nama:
`10.png`, `10 (2).png`, `10 - Copy.png`, `10 - Copy (2).png`, `10 - Copy (3).png`,
`10 - Copy - Copy.png`, `10 - Copy (2) - Copy.png`.

Pengecekan MD5 (lihat `Kodingan/outputs/manifests/full_manifest.csv`) menunjukkan kedelapan file
ini sebenarnya hanya **4 gambar unik**, masing-masing digandakan tepat 1 kali dengan pola
penamaan khas hasil *copy-paste* Windows Explorer ("- Copy", "- Copy (2)"):

| Gambar unik | Nama file kembar |
|---|---|
| A | `10 (2) - Copy.png` = `10 (2).png` |
| B | `10 - Copy (2) - Copy.png` = `10 - Copy (2).png` |
| C | `10 - Copy (3).png` = `10.png` |
| D | `10 - Copy - Copy.png` = `10 - Copy.png` |

Kemungkinan besar A, B, C, D adalah 4 slice CT berbeda dari studi/kasus #10 yang sama, yang
masing-masing kebetulan tergandakan satu kali saat proses pengumpulan data. Baik dedup MD5
maupun pengelompokan `case_key` (nomor kasus dari nama file) sama-sama menangkap pola ini --
lihat `Kejanggalan Dataset/03_duplikasi_dalam_satu_dataset.md`.
""",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # 3) Augmentasi sintetis (Das) yang ternyata copy asli tanpa augmentasi
    # ------------------------------------------------------------------ #
    d3 = OUT_ROOT / "03_Augmentasi_Sintetis_Menyamar_Data_Asli"
    das = df[df["is_synthetic_source"]]
    real = df[~df["is_synthetic_source"]]
    for i, name in enumerate(["Normal case (139).jpg", "Normal case (242).jpg"], start=1):
        das_row = das[das["filename"] == name]
        real_row = real[real["filename"] == name]
        if len(das_row) and len(real_row):
            sub_dir = d3 / f"contoh_{i:02d}_{name.replace(' ', '_').replace('.jpg','')}"
            copy_files(pd.concat([das_row.iloc[[0]], real_row.iloc[[0]]]), sub_dir)
    (d3 / "README.md").write_text(
        """# Augmentasi Sintetis (Das) yang Ternyata Salinan Asli Tanpa Augmentasi

Dataset "IQ-OTHNCCD Lung Cancer Dataset (Augmented) (Subhajeet Das)" mengklaim seluruh isinya
adalah hasil augmentasi. Audit menemukan **1.054 dari 3.609 file (~29%) byte-identical** (MD5
sama persis) dengan citra yang sudah ada di dataset lain -- artinya sebagian file di dalamnya
BUKAN hasil augmentasi baru, melainkan salinan mentah dari citra asli.

Setiap subfolder di sini berisi 2 file yang **identik secara MD5**:
- Satu dari `IQ-OTHNCCD Lung Cancer Dataset (Augmented) (Subhajeet Das)`
- Satu lagi dari `CT Scan Images for Lung Cancer (Dishan rathi20)` (yang sendiri adalah
  gabungan Mohamed Hany + IQ-OTH/NCCD, lihat dokumen 01)

Bandingkan kedua file di setiap subfolder -- keduanya akan terlihat identik. Lihat
`Kejanggalan Dataset/02_dataset_augmentasi_sintetis.md`.
""",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # 4) Data tanpa label yang terduplikasi
    # ------------------------------------------------------------------ #
    d4 = OUT_ROOT / "04_Data_Tanpa_Label_Terduplikasi"
    grp891 = dup[dup["group_id"] == 891].drop_duplicates("source_dataset")
    copy_files(grp891, d4 / "contoh_01_test_cases_identik")
    (d4 / "README.md").write_text(
        """# Data "Test cases" Tanpa Label yang Terduplikasi

Folder `contoh_01_test_cases_identik/` berisi 1 file yang sama persis (MD5 identik), diklaim
sebagai bagian dari 2 dataset "Test cases" berbeda:
- `IQ-OTHNCCD - Lung Cancer Dataset (Aditya Mahimkar)/Test cases/`
- `CT Scan Images for Lung Cancer (Dishan rathi20)/Test cases/`

Kedua folder Test cases ini (masing-masing 197 file) terbukti **100% identik** satu sama lain
(197/197 MD5 sama) dan **tidak memiliki label ground-truth** sama sekali, sehingga tidak dipakai
untuk training/evaluasi. Lihat `Kejanggalan Dataset/04_data_tanpa_label_dan_kualitas_gambar.md`.
""",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # 5) Bug deteksi otomatis: hash collision palsu (BUKAN salah diagnosis asli)
    # ------------------------------------------------------------------ #
    d5 = OUT_ROOT / "05_Kegagalan_Deteksi_Otomatis_Hash_Collision"
    p1 = df[df["path"].str.contains("vertical_flip \\(14\\).jpg", regex=True) & df["path"].str.contains("Benign")]
    p2 = df[df["path"].str.contains("vertical_flip \\(96\\).jpg", regex=True) & df["path"].str.contains("Normal")]
    copy_files(pd.concat([p1, p2]), d5 / "contoh_01_phash_collision_benign_vs_normal")
    (d5 / "README.md").write_text(
        """# Kegagalan Deteksi Otomatis: Tabrakan Hash Palsu (BUKAN Bukti Salah Diagnosis)

**Penting: folder ini BUKAN contoh gambar yang salah label/salah diagnosis.** Ini adalah bukti
visual dari sebuah *false positive* yang sempat terjadi saat mengembangkan alat audit ini sendiri.

`vertical_flip (14).jpg` (folder "Benign cases") dan `vertical_flip (96).jpg` (folder "Normal
cases") -- keduanya dari dataset augmentasi sintetis Subhajeet Das -- memiliki **perceptual hash
64-bit yang identik persis**, padahal MD5-nya berbeda (isi piksel sungguh berbeda; keduanya
adalah hasil flip vertikal dari 2 pasien yang berbeda). Buka dan bandingkan kedua file di folder
ini secara visual -- Anda akan melihat keduanya memang gambar yang berbeda, bukan gambar yang
sama dengan label tertukar.

Tabrakan hash inilah yang sempat (sebelum diperbaiki) menyebabkan seluruh kelas Benign hilang
dari pool kanonik karena tergabung secara keliru dengan kelas Normal melalui rantai transitif.
Kronologi lengkap ada di `Kejanggalan Dataset/06_bug_metodologi_yang_ditemukan_dan_diperbaiki.md`.
""",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # 6) Kandidat "salah label" -- honest report: strict search found only
    #    false-positive fan-out patterns, not genuine mislabels
    # ------------------------------------------------------------------ #
    d6 = OUT_ROOT / "06_Kandidat_Kemungkinan_Salah_Label_TIDAK_TERVERIFIKASI"
    ex1 = df[df["filename"] == "Normal case (8).jpg"]
    ex1 = ex1[ex1["source_dataset"].str.contains("rathi20")]
    ex2 = df[(df["filename"].isin(["Malignant case (89).jpg", "Malignant case (100).jpg"]))
             & (df["source_dataset"].str.contains("rathi20"))]
    copy_files(pd.concat([ex1, ex2]), d6 / "contoh_01_fan_out_palsu")
    (d6 / "README.md").write_text(
        """# Kandidat Kemungkinan Salah Label -- HASIL: TIDAK TERBUKTI (Penting Dibaca)

Sesuai permintaan, dilakukan pencarian tambahan khusus untuk kasus "gambar yang harusnya
no-cancer tapi terlabel cancer (atau sebaliknya)" -- dengan kriteria yang lebih longgar dari
metode utama (perceptual-hash jarak <=4 DAN average-hash jarak <=4 DAN resolusi identik),
dibatasi hanya pada pasangan lintas label (Normal vs Malignant, Normal vs Benign).

**Hasil: audit utama (MD5 exact + perceptual-hash exact) yang dipakai untuk keputusan pool
kanonik menemukan 0 (nol) kasus gambar identik dengan label bertentangan** -- lihat
`audit_summary.json`, field `label_conflict_groups: 0`. Ini kabar baik: tidak ada bukti
mislabel yang solid pada data yang dipakai training.

Pencarian tambahan yang lebih longgar (khusus untuk permintaan ini) memang menemukan 405
pasangan "kandidat", TETAPI setelah diperiksa visual secara langsung, pasangan-pasangan ini
adalah **false positive** dari masalah yang sama seperti di folder 05 (citra CT paru-paru
punya struktur visual yang mirip secara umum -- FOV melingkar, kontras serupa -- sehingga
hash tidak cukup diskriminatif).

Contoh di `contoh_01_fan_out_palsu/`: `Normal case (8).jpg` "cocok" dengan **5 gambar
Malignant berbeda sekaligus** (`Malignant case (89/90/100/101/102).jpg`) pada pencarian
longgar ini. Satu gambar tidak mungkin secara valid identik dengan 5 gambar lain yang
berbeda-beda -- pola "satu lawan banyak" ini adalah tanda klasik tabrakan hash, bukan
duplikat sungguhan. **Buka dan bandingkan sendiri gambar-gambar di folder ini** -- akan
terlihat jelas keduanya adalah pasien yang berbeda (bentuk tubuh, pola pembuluh darah paru,
dan anatomi tulang rusuk berbeda).

**Kesimpulan jujur untuk skripsi:** tidak ditemukan bukti kuat adanya citra yang salah
label/salah diagnosis pada dataset yang dipakai. Kalaupun ada mislabel yang sesungguhnya
tersembunyi di salah satu dari 7 dataset asli, metode yang tersedia (hash-based) tidak
mampu mendeteksinya secara andal pada domain citra CT paru-paru -- ini didokumentasikan
sebagai keterbatasan (limitation) penelitian, bukan diklaim sebagai temuan positif.
""",
        encoding="utf-8",
    )

    print("Selesai. Struktur folder yang dibuat:")
    for p in sorted(OUT_ROOT.rglob("*")):
        if p.is_dir():
            n = len(list(p.glob("*.*"))) - (1 if (p / "README.md").exists() else 0)
            print(f"  {p.relative_to(OUT_ROOT)}  ({n} file gambar)" if n >= 0 else f"  {p.relative_to(OUT_ROOT)}")


if __name__ == "__main__":
    main()
