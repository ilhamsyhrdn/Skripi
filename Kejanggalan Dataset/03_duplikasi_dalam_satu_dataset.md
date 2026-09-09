# Kejanggalan #3 — Duplikasi di Dalam Satu Dataset (Bukan Lintas Dataset)

## Temuan A — Duplikat hasil copy-paste Windows Explorer

Pada folder `Chest CT-Scan images Dataset (Mohamed Hany)/test/normal/` dan folder serupa pada
dataset Dishan Rathi, ditemukan nama file dengan pola khas hasil *copy-paste* Windows Explorer:

```
10.png
10 (2).png
10 - Copy.png
10 - Copy (2).png
10 - Copy (2) - Copy.png
10 - Copy (3).png
10 - Copy - Copy.png
```

Tujuh nama file berbeda untuk **kemungkinan satu gambar asli yang sama**, diperbanyak berkali-kali
dengan cara duplikasi manual (bukan augmentasi/transformasi gambar — nama file "- Copy" adalah
penanda otomatis Windows saat sebuah file di-*paste* ke folder yang sama). Pola ini terjadi berulang
untuk banyak indeks kasus (11, 12, dst.) pada folder yang sama, membuat kelas "normal" tampak jauh
lebih besar dari jumlah pasien/slice unik yang sebenarnya.

## Temuan B — Banyak crop/slice per ID pasien pada dataset bergaya LIDC

Dataset Mohamed Hany, Dishan Rathi, dan Waseim Nagah Hennes memakai skema penamaan
`<ID 6-digit> (<varian>).png`, contoh: `000005 (3).png`, `000005 (9).png`. ID 6 digit di depan sangat
mungkin merupakan ID pasien/studi (mengikuti konvensi ekspor DICOM yang umum dipakai LIDC-IDRI),
sedangkan angka dalam kurung adalah varian crop/slice dari studi yang sama. Sebuah ID pasien dapat
memiliki puluhan varian crop pada satu folder kelas yang sama.

## Dampak bila tidak ditangani

Baik Temuan A maupun B sama-sama berarti: **banyak file gambar berbeda nama, tapi berasal dari
sumber piksel/pasien yang sama**. Split train/test acak per-file (bukan per-pasien/per-grup) akan
memecah varian-varian dari sumber yang sama ke sisi train dan test sekaligus — jenis *patient-level
leakage* yang secara eksplisit pernah ditemukan pada proyek sebelumnya di repository ini (lihat
commit `623189f`: pasangan "case (93)" vs "case (94)" dengan *cosine similarity* hingga 1.0000
ternyata terpisah antara train dan holdout).

## Penanganan pada skripsi ini

Dua mekanisme independen dipakai bersamaan:

1. **`case_key`** — diekstrak dari nama file (ID di depan, atau angka dalam kurung setelah nama
   kelas) dan dipakai untuk menyatukan file-file dengan bukti nama-file yang sama menjadi satu grup
   sebelum deduplikasi (lihat `extract_case_key()` di `build_manifest.py`). Ini yang menyatukan
   ketujuh varian "10*.png" di atas menjadi satu grup, tanpa bergantung pada kemiripan visual yang
   ternyata tidak reliabel untuk citra CT paru-paru (lihat dokumen 06).
2. **`split_group`** — pada tahap pembuatan split (`Kodingan/src/audit/make_splits.py`), setiap
   gambar kanonik dikelompokkan lebih lanjut berdasarkan `(sumber, folder, nomor_kasus // 5)` sebagai
   margin keamanan tambahan, sehingga nomor kasus yang berdekatan pada folder yang sama (yang secara
   audit terbukti bisa jadi crop/slice dari studi yang sama) selalu berada pada split yang sama.
