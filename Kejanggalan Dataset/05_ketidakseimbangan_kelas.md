# Kejanggalan #5 — Ketidakseimbangan Kelas pada Pool Data Bersih

Setelah deduplikasi menyeluruh (lihat dokumen 01-03), pool data kanonik yang benar-benar unik hanya
berjumlah **894 gambar**, dengan distribusi:

| Kelas | Jumlah gambar | Proporsi |
|---|---:|---:|
| Malignant | 665 | 74,4% |
| Normal | 136 | 15,2% |
| Benign | 93 | 10,4% |

Rasio ketidakseimbangan (kelas terbesar / terkecil) = **7,15 : 1**.

## Mengapa ini terjadi

Kelas Malignant mendapat kontribusi data riil dari **dua sumber independen** yang tidak saling
tumpang tindih: (1) IQ-OTH/NCCD (kelas Malignant tanpa info subtipe, 416 gambar unik setelah dedup)
dan (2) dataset bergaya Mohamed Hany/Nafees Imtiaz yang memberi label subtipe eksplisit
(Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma — total 249 gambar unik setelah
dedup). Sebaliknya, kelas Benign dan Normal pada dasarnya **hanya berasal dari satu sumber asli**
(IQ-OTH/NCCD) yang diunggah ulang berkali-kali tanpa menambah keragaman pasien nyata — sehingga
setelah deduplikasi, jumlahnya tetap kecil dan mendekati jumlah pasien asli pada studi Al-Yasriy et
al.

## Dampak terhadap desain model

Ketidakseimbangan sebesar ini, ditambah ukuran dataset yang kecil secara keseluruhan (894 gambar
untuk 3 kelas), adalah alasan utama mengapa metodologi skripsi ini secara sengaja menggabungkan tiga
strategi sekaligus (bukan hanya satu):

1. **Transfer learning** (bobot ImageNet EfficientNet-B0) — supaya model tidak perlu mempelajari
   fitur visual dasar dari nol dengan data sekecil ini (lih. Kim et al., 2022, tinjauan pustaka).
2. **Data augmentation** yang diterapkan hanya pada sisi *train* — untuk memperbesar variasi efektif
   tanpa menambah risiko kebocoran data (lih. Chlap et al., 2021).
3. **Class-weighted loss function** (`Kodingan/src/training/train_cv.py`, `class_weights_from_df()`)
   — bobot loss per kelas dihitung berbanding terbalik dengan frekuensinya pada setiap fold training,
   supaya model tidak hanya belajar memprediksi kelas mayoritas (Malignant) untuk memaksimalkan
   akurasi mentah.
4. **Ensemble model** (5-fold cross-validation × 2 arsitektur) — dengan data bersih yang terbatas dan
   kelas minoritas yang sangat kecil (Benign hanya 93 gambar, ±15-19 per fold validasi), satu model
   tunggal sangat rentan terhadap varians tinggi antar-run. Menggabungkan prediksi beberapa model
   yang dilatih pada subset data dan/atau arsitektur yang berbeda menstabilkan estimasi probabilitas
   akhir dan pada gilirannya meningkatkan recall & akurasi secara lebih andal dibanding
   mengandalkan satu model saja — sejalan dengan pendekatan Shi et al. (2022) dan Saha et al. (2024)
   yang dilampirkan pada tinjauan pustaka.
5. Pelaporan metrik **per-kelas** (terutama *recall* kelas Malignant/"cancer recall") sebagai metrik
   utama, bukan hanya akurasi keseluruhan — karena pada dataset timpang seperti ini, akurasi
   keseluruhan yang tinggi dapat dicapai dengan mudah hanya dengan selalu menebak kelas mayoritas,
   tanpa benar-benar berguna secara klinis untuk menangkap kasus Benign/Normal.
