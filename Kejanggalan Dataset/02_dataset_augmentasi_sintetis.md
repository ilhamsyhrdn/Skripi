# Kejanggalan #2 — Dataset "Augmented" (Subhajeet Das) adalah Data Sintetis dari Dataset Lain

## Temuan

`IQ-OTHNCCD Lung Cancer Dataset (Augmented) (Subhajeet Das)` berisi 3.609 gambar (Benign 1.200,
Malignant 1.201, Normal 1.208) — jauh lebih banyak dari dataset IQ-OTH/NCCD asli (Benign 120,
Malignant 561, Normal 416).

Bukti bahwa isinya adalah hasil augmentasi (bukan data pasien baru):

1. **Nama file identik dengan dataset asli untuk sebagian file** — mis. `Bengin case (1).jpg` muncul
   persis sama di folder Al-Yasriy maupun di folder Das.
2. **Nama file lain secara eksplisit menandakan augmentasi**, contoh yang ditemukan langsung di
   audit: `vertical_flip (14).jpg`, `vertical_flip (96).jpg` — jelas hasil operasi *flip* terprogram,
   bukan slice CT baru.
3. Pengecekan hash terpisah (lihat `audit_summary.json`, field
   `synthetic_images_exact_byte_match_to_real_pool` dan `synthetic_images_exact_phash_match_to_real_pool`)
   menemukan **1.097 dari 3.609 gambar Das (≈30%) byte-identical** dengan gambar yang sudah ada di
   pool data asli, dan **1.077 gambar lagi (≈30%)** cocok persis pada perceptual-hash (kemungkinan
   hasil kompresi ulang dari gambar asli yang sama). Total ≈60% dari dataset ini terbukti langsung
   sebagai turunan/duplikat dari data yang sudah ada di 6 dataset lain, dan sisanya yang tidak
   terdeteksi hash (rotasi/flip mengubah hash secara signifikan) kemungkinan besar juga merupakan
   augmentasi dari gambar yang sama, hanya tidak terbukti lewat hash.

## Dampak bila tidak ditangani

Data augmentasi seperti ini pada dasarnya adalah **transformasi geometris/intensitas dari gambar
yang sama** (persis seperti teknik `data augmentation` yang dijelaskan Chlap et al., 2021, pada
tinjauan pustaka skripsi ini). Jika dataset ini digabung ke pool data mentah lalu displit train/test
secara acak, versi *flip* dari sebuah gambar sangat mungkin masuk ke **train** sementara versi
aslinya (atau versi *flip* lainnya) masuk ke **test** — model akan "mengenali" pola augmentasi dari
gambar yang sama yang sudah pernah dilihatnya, menyebabkan estimasi akurasi test yang optimis secara
tidak jujur.

## Penanganan pada skripsi ini

Dataset ini **dikeluarkan sepenuhnya** dari pool kanonik (`is_synthetic_source=True` di
`full_manifest.csv`). Augmentasi tetap dipakai pada skripsi ini — tetapi dilakukan **on-the-fly saat
training** (lihat `Kodingan/src/data/dataset.py`, fungsi `build_transforms()`) hanya pada gambar yang
sudah pasti berada di sisi *train* setelah split dilakukan, bukan diambil dari file augmentasi siap
pakai yang sumber pasangannya (asli vs hasil augmentasi) tidak diketahui berada di split mana.

Sebagai catatan metodologis penting: dataset Das ini juga sempat **merusak proses deduplikasi**
ketika awalnya diikutsertakan dalam pencocokan hash — lihat
`06_bug_metodologi_yang_ditemukan_dan_diperbaiki.md` untuk kronologinya.
