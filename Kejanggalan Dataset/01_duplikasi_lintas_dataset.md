# Kejanggalan #1 — Duplikasi Lintas Dataset (7 Folder Kaggle Bukan 7 Sumber Independen)

## Temuan

Ketujuh dataset yang diunduh dari Kaggle dan diletakkan di `D:\skripsi\Dataset\` sebenarnya berasal
dari hanya **2 sumber data asli** yang diunggah ulang berkali-kali oleh uploader Kaggle yang berbeda,
kadang dengan nama folder/label yang berbeda sehingga terlihat seperti dataset baru.

### Bukti A — IQ-OTH/NCCD diunggah ulang 3 kali secara identik

| Dataset (uploader Kaggle) | Benign | Malignant | Normal | Catatan |
|---|---:|---:|---:|---|
| The IQ-OTHNCCD lung cancer dataset (**Hamdalla F. Al-Yasriy** — penulis asli) | 120 | 561 | 416 | Sumber asli |
| IQ-OTHNCCD - Lung Cancer Dataset (**Aditya Mahimkar**) | 120 | 561 | 416 | **Angka identik persis** ke sumber asli |
| Lung cancer dataset (IQ-OTHNCCD) (**Waseim Nagah Hennes**) | 120 | 1339 | 614 | Benign identik; Malignant/Normal lebih banyak karena tercampur dengan sumber lain (lihat Bukti B) |

Nama file pada folder Al-Yasriy ("Bengin case (1).jpg", "Malignant case (1).jpg", dst.) muncul
**identik** pada folder Mahimkar. Pengecekan hash MD5 mengonfirmasi ribuan file **byte-identical**
antar folder ini (lihat kolom `md5` pada `full_manifest.csv`, atau grup-grup besar di
`duplicate_groups.csv` yang kolom `group_sources`-nya memuat lebih dari satu nama dataset).

### Bukti B — Dataset "Chest CT-Scan images" (mohamedhanyyy) tertanam di 2 dataset lain

Dataset ini (1.000 gambar: adenocarcinoma/large-cell/squamous/normal dengan penamaan folder
bertahap TNM, mis. `adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib`) adalah dataset Kaggle terkenal yang
juga dipakai pada jurnal COGITO Smart Journal (Sandag & Kabo, 2024) yang dilampirkan pada skripsi
ini. Audit menemukan:

| Perbandingan | Jumlah file sumber A | Jumlah file sumber B | Overlap MD5 (identik persis) |
|---|---:|---:|---:|
| Mohamed Hany vs **Dishan Rathi** | 1.000 | 2.274 | **847 file (≈85% dari Hany)** |
| Mohamed Hany vs **MD. Nafees Imtiaz** (via subtype folder) | 1.000 | 1.535 | file dengan nama identik `adenocarcinoma (N).png` dst., jumlah kelas subtipe sama |

Dataset "Dishan Rathi" ternyata adalah **gabungan** dataset Mohamed Hany (subtipe kanker + normal)
**dengan** dataset IQ-OTH/NCCD (kelas Benign/Malignant terpisah) yang di-*repackage* ulang menjadi
satu folder besar dengan struktur folder train/valid/test yang terlihat baru, padahal isinya adalah
gabungan dua dataset yang sudah ada.

### Bukti C — folder "Test cases" identik 100% antara 2 dataset

`IQ-OTHNCCD - Lung Cancer Dataset (Aditya Mahimkar)/Test cases` (197 file) dan
`CT Scan Images for Lung Cancer (Dishan rathi20)/Test cases` (197 file) memiliki
**197/197 MD5 identik** — file yang sama persis, sama-sama tanpa label ground-truth.

## Dampak bila tidak ditangani

Jika ketujuh folder digabung langsung lalu dilakukan split train/test/valid secara acak per gambar
(pendekatan naif yang umum dilakukan), gambar yang **identik** akan sangat mungkin muncul di kedua
sisi split (satu salinan di train, satu salinan lainnya di test) — model secara efektif "menghafal"
gambar test karena sudah pernah melihat salinan pixel-identik-nya saat training. Ini persis jenis
*data leakage* yang menyebabkan angka akurasi 78-79% pada proyek sebelumnya di repository ini
ternyata tidak valid (lihat commit `01837e2`, akurasi jujur setelah dibersihkan turun ke 69%).

## Penanganan pada skripsi ini

- Deduplikasi dilakukan pada level **hash gambar (MD5 exact + perceptual-hash exact)**, bukan pada
  level "nama dataset", sehingga duplikat lintas sumber manapun akan otomatis terdeteksi tanpa perlu
  daftar manual dataset mana yang duplikat dari mana.
- Untuk setiap grup duplikat, hanya **1 representasi** (resolusi tertinggi) yang dipertahankan pada
  `canonical_pool.csv`; sisanya dibuang dari pool training/evaluasi.
- Folder "Test cases" yang tanpa label dibuang seluruhnya (lihat dokumen 04).

Lihat kode: `Kodingan/src/audit/build_manifest.py`, fungsi `discover_images()`, `fingerprint()`, dan
bagian "Grouping exact duplicates (MD5)" / "Grouping exact perceptual-hash matches".
