# Kejanggalan #4 — Data Tanpa Label & Variasi Kualitas/Resolusi Gambar

## Data tanpa label ground-truth ("Test cases")

Dua dataset (`IQ-OTHNCCD - Lung Cancer Dataset (Aditya Mahimkar)` dan
`CT Scan Images for Lung Cancer (Dishan rathi20)`) masing-masing memiliki folder `Test cases`
berisi 197 gambar **tanpa subfolder kelas** (tidak ada label Benign/Malignant/Normal). Audit
mengonfirmasi kedua folder ini **100% identik** (197/197 MD5 sama) — jadi hanya ada 197 gambar unik
tanpa label, diklaim dua kali sebagai bagian dari dua dataset "berbeda".

Pengecekan tambahan: tidak ada satu pun dari 197 gambar ini yang cocok (MD5) dengan gambar berlabel
mana pun di 5 dataset lain, jadi keberadaannya **tidak** membocorkan label ke pool data berlabel —
tetapi karena tidak ada ground truth, gambar ini tidak dapat dipakai untuk supervised training/
evaluation dan **dikeluarkan seluruhnya** dari pool kanonik (kolom `is_labeled=False` pada
`full_manifest.csv`).

## Variasi resolusi antar sumber

| Sumber | Jumlah kombinasi (lebar,tinggi) berbeda | Interpretasi |
|---|---:|---|
| Al-Yasriy (IQ-OTH/NCCD asli) | 5 | Resolusi sudah distandardisasi oleh penulis dataset |
| Mahimkar (IQ-OTH/NCCD reupload) | 6 | Sama, sumber identik |
| Mohamed Hany | 798 | Hampir setiap gambar unik ukurannya |
| Dishan Rathi | 803 | idem |
| Nafees Imtiaz | 798 | idem |
| Hennes | 795 | idem |

Dataset yang bersumber dari gaya crop nodul (Hany/Rathi/Nafees/Hennes) memiliki bounding-box crop
yang berbeda-beda per gambar sehingga resolusi hampir selalu unik — konsisten dengan asumsi bahwa
data ini berasal dari proses ekstraksi ROI otomatis per nodul (mirip gaya LIDC-IDRI), bukan dari
ekspor slice CT utuh yang seragam.

## Penanganan pada skripsi ini

- Folder "Test cases" dibuang (lihat di atas).
- Seluruh gambar di-*resize* ke ukuran tetap 224×224 piksel saat training/inferensi
  (`Kodingan/src/data/dataset.py`, `build_transforms()`), sesuai kebutuhan input EfficientNet-B0,
  sehingga variasi resolusi mentah tidak berpengaruh terhadap arsitektur model. Tidak ditemukan file
  gambar yang korup/tidak terbaca (`corrupt_files: 0` pada `audit_summary.json`) — seluruh 12.882
  file berhasil dibuka dengan PIL tanpa error.
