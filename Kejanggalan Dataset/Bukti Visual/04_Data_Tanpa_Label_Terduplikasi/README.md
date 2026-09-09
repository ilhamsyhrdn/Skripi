# Data "Test cases" Tanpa Label yang Terduplikasi

Folder `contoh_01_test_cases_identik/` berisi 1 file yang sama persis (MD5 identik), diklaim
sebagai bagian dari 2 dataset "Test cases" berbeda:
- `IQ-OTHNCCD - Lung Cancer Dataset (Aditya Mahimkar)/Test cases/`
- `CT Scan Images for Lung Cancer (Dishan rathi20)/Test cases/`

Kedua folder Test cases ini (masing-masing 197 file) terbukti **100% identik** satu sama lain
(197/197 MD5 sama) dan **tidak memiliki label ground-truth** sama sekali, sehingga tidak dipakai
untuk training/evaluasi. Lihat `Kejanggalan Dataset/04_data_tanpa_label_dan_kualitas_gambar.md`.
