# HALAMAN JUDUL

**OPTIMASI *TRANSFER LEARNING* EFFICIENTNET-B0 UNTUK KLASIFIKASI KANKER PARU-PARU PADA CITRA *COMPUTED TOMOGRAPHY* (CT) MENGGUNAKAN *FINE-TUNING*, *DATA AUGMENTATION*, DAN *ENSEMBLE MODEL***

**SKRIPSI**

diajukan untuk menempuh ujian sarjana
pada Fakultas Matematika dan Ilmu Pengetahuan Alam
Universitas Padjadjaran

MUHAMMAD ILHAMSYAH RIDWAN
NPM 140810220059

UNIVERSITAS PADJADJARAN
FAKULTAS MATEMATIKA DAN ILMU PENGETAHUAN ALAM
PROGRAM STUDI TEKNIK INFORMATIKA
SUMEDANG
2026

---

# LEMBAR PENGESAHAN

**SKRIPSI**

**OPTIMASI *TRANSFER LEARNING* EFFICIENTNET-B0 UNTUK KLASIFIKASI KANKER PARU-PARU PADA CITRA *COMPUTED TOMOGRAPHY* (CT) MENGGUNAKAN *FINE-TUNING*, *DATA AUGMENTATION*, DAN *ENSEMBLE MODEL***

***OPTIMIZATION OF EFFICIENTNET-B0 TRANSFER LEARNING FOR LUNG CANCER CLASSIFICATION ON COMPUTED TOMOGRAPHY (CT) IMAGES USING FINE-TUNING, DATA AUGMENTATION, AND ENSEMBLE MODEL***

Telah dipersiapkan dan disusun oleh

MUHAMMAD ILHAMSYAH RIDWAN
NPM 140810220059

Telah dipertahankan di depan Tim Penguji
pada tanggal ……………………

Susunan Tim Penguji

| No | Nama | Peran | NIP |
|---|---|---|---|
| 1 | …………………………………… | Ketua Tim Penguji | …………………………………… |
| 2 | Erick Paulus, S.Si, M.Kom. | Pembimbing | 19820318 200604 1 001 |
| 3 | Dr. Asep Sholahuddin, MT. | Co-Pembimbing | 19670403 199303 1 002 |
| 4 | …………………………………… | Penguji | …………………………………… |
| 5 | …………………………………… | Penguji | …………………………………… |
| 6 | …………………………………… | Penguji | …………………………………… |

---

## KATA PENGANTAR

Puji dan syukur penulis panjatkan ke hadirat Tuhan Yang Maha Esa yang telah memberikan rahmat dan karunia-Nya sehingga penulis dapat menyelesaikan penyusunan skripsi yang berjudul “**Optimasi *Transfer Learning* EfficientNet-B0 untuk Klasifikasi Kanker Paru-Paru pada Citra *Computed Tomography* (CT) Menggunakan *Fine-Tuning*, *Data Augmentation*, dan *Ensemble Model***” sebagai salah satu syarat menempuh ujian sarjana pada Program Studi S-1 Teknik Informatika Fakultas Matematika dan Ilmu Pengetahuan Alam Universitas Padjadjaran.

Dalam proses penyusunan dan penulisan skripsi ini tidak terlepas dari bantuan, bimbingan, serta dukungan dari berbagai pihak. Oleh karena itu, dalam kesempatan ini penulis mengucapkan terima kasih kepada Bapak Erick Paulus, S.Si, M.Kom., sebagai pembimbing utama, dan Bapak Dr. Asep Sholahuddin, MT., sebagai pembimbing pendamping yang telah meluangkan waktu dan pikirannya sehingga penulis dapat menyelesaikan skripsi ini. Ucapan terima kasih juga diberikan kepada keluarga penulis yang selalu memberikan motivasi dan doa yang menjadi pendorong dalam penyelesaian skripsi ini. Penulis juga mengucapkan terima kasih sebanyak-banyaknya kepada:

1. Prof. Dr. Desi Harneti Putri Huspa, S.Si., M.Si., selaku Dekan Fakultas Matematika dan Ilmu Pengetahuan Alam Universitas Padjadjaran.
2. Dr. Afrida Helen, S.T., M.Kom., selaku Kepala Departemen Ilmu Komputer Fakultas Matematika dan Ilmu Pengetahuan Alam Universitas Padjadjaran.
3. Dr. Akmal, S.Si., M.T., selaku Ketua Program Studi S-1 Teknik Informatika Fakultas Matematika dan Ilmu Pengetahuan Alam Universitas Padjadjaran.
4. Bapak dan Ibu dosen penguji yang telah berkenan menjadi dosen penguji pada seminar hasil dan sidang skripsi, serta memberikan masukan dan saran untuk penyempurnaan skripsi ini.
5. Dosen-dosen Program Studi S-1 Teknik Informatika Universitas Padjadjaran yang telah mengajar dan memberikan ilmu kepada penulis selama masa perkuliahan yang membawa penulis pada posisi sekarang ini.
6. Rekan-rekan mahasiswa Program Studi S-1 Teknik Informatika Universitas Padjadjaran yang telah memberikan bantuan, dukungan, dan tempat berdiskusi selama penelitian ini berlangsung.

Akhir kata, penulis berharap semoga skripsi ini dapat bermanfaat bagi pembaca.

Jatinangor, September 2026

Penulis

---

## ABSTRAK

Kanker paru-paru merupakan penyebab utama kematian akibat kanker, sedangkan pembacaan citra *Computed Tomography* (CT) secara manual memakan waktu. Penelitian ini bertujuan mengoptimasi *transfer learning* EfficientNet-B0 untuk mengklasifikasikan citra CT paru-paru sebagai *Malignant*, *Benign*, atau Normal.

Dalam penelitian eksperimental ini, 2.811 citra CT dari Kaggle dan LIDC-IDRI dipakai untuk *fine-tuning* dua fase dengan *data augmentation*, dilanjutkan *ensemble stacking* bersama ResNet50 dan evaluasi pada 442 citra uji terpisah.

Model akhir mencapai akurasi 82,35%, *macro-F1* 0,7228, dan *cancer recall* 91,00%, dengan kontribusi terbesar dari *stacking*. Model ini menunjukkan potensi sebagai alat bantu skrining, meskipun kelas *Benign* masih sulit dikenali.

**Kata Kunci**: *Data Augmentation*, EfficientNet-B0, *Ensemble Model*, *Fine-Tuning*, Kanker Paru-Paru

---

## ABSTRACT

Lung cancer is the leading cause of cancer death, while manual reading of Computed Tomography (CT) images is time-consuming. This research aimed to optimize EfficientNet-B0 transfer learning to classify lung CT images as Malignant, Benign, or Normal.

In this experimental study, 2,811 CT images from Kaggle and LIDC-IDRI were used for two-phase fine-tuning with data augmentation, followed by stacking with ResNet50 and evaluation on 442 held-out images.

The final model achieved 82.35% accuracy, 0.7228 macro-F1, and 91.00% cancer recall, with stacking contributing most. It showed potential as a screening aid, although the Benign class remained difficult.

**Keywords**: Data Augmentation, EfficientNet-B0, Ensemble Model, Fine-Tuning, Lung Cancer
