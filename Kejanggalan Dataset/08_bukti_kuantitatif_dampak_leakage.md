# Bukti Kuantitatif — Seberapa Besar Data Leakage Menggelembungkan Akurasi

Dokumen ini melaporkan eksperimen pembanding yang **sengaja mereplikasi kesalahan umum**
(menggabungkan seluruh dataset lalu displit train/test secara acak per gambar, tanpa
deduplikasi atau pengelompokan pasien) untuk mengukur secara konkret seberapa besar dampaknya,
dibandingkan pipeline final yang dipakai pada skripsi ini. Kode: `Kodingan/src/audit/naive_leaky_baseline.py`.

## Setup eksperimen

| | **Baseline Naif (bocor)** | **Pipeline Final (bersih)** |
|---|---|---|
| Data sumber | 8.879 gambar berlabel (SEMUA duplikat & near-duplikat lintas dataset diikutsertakan, hanya dataset sintetis Das yang dibuang) | 894 gambar unik (dedup penuh, lihat dokumen 01-07) |
| Metode split | Random split per **gambar**, stratifikasi kelas saja (`train_test_split`, 70/15/15) | Held-out test 18% + 5-fold `StratifiedGroupKFold` per **grup duplikat & rumpun kasus** |
| Model | 1× EfficientNet-B0 (transfer learning + fine-tuning 2 fase, identik dengan pipeline final) | Ensemble 10 model (5× EfficientNet-B0 + 5× ResNet50, soft-voting) |
| Ukuran test set | 1.332 gambar | 165 gambar |

## Bukti kebocoran pada baseline naif (diverifikasi, bukan diasumsikan)

Menggunakan `group_id` hasil audit deduplikasi (dokumen 01-03) sebagai **ground truth independen**
untuk memeriksa split random-per-gambar:

- **99,85% dari gambar test pada baseline naif memiliki grup duplikat yang juga muncul di
  train-nya sendiri.**
- **99,55% dari gambar validation** juga demikian.

Artinya hampir seluruh gambar "test" pada baseline naif sebenarnya adalah salinan/near-duplikat
dari gambar yang sudah dilihat model saat training — validasi ini pada dasarnya mengukur seberapa
baik model **menghafal**, bukan seberapa baik model **generalisasi**.

## Hasil

| Metrik | Baseline Naif (bocor) | Pipeline Final (bersih, ensemble) | Selisih |
|---|---:|---:|---:|
| Akurasi | **94,5%** | 85,5% | **+9,0 poin persentase (palsu)** |
| Macro-F1 | **0,881** | 0,695 | **+0,186 (palsu)** |
| Recall Malignant (cancer recall) | 97,6% | 92,6% | +5,0 poin (palsu) |

Sumber angka: `Kodingan/outputs/reports/naive_leaky_baseline.json` vs
`Kodingan/outputs/reports/ensemble_evaluation.json`.

## Kesimpulan

Angka pada kolom "Baseline Naif" terlihat lebih tinggi dan lebih meyakinkan di atas kertas —
tepat seperti yang terjadi pada proyek sebelumnya di repository ini (commit `01837e2`: akurasi
78-79% pada data bocor turun menjadi 69% setelah dibersihkan) — namun angka tersebut **tidak
mengukur kemampuan generalisasi model yang sesungguhnya**, karena hampir seluruh "data test"-nya
sudah pernah dilihat modelnya dalam bentuk salinan/near-duplikat saat training.

Angka pada kolom "Pipeline Final" lebih rendah tetapi **jujur**: diuji pada 165 citra yang benar-
benar independen (grup duplikat maupun rumpun nomor kasusnya tidak pernah menyentuh data training
manapun selama 5-fold cross-validation). Inilah yang dilaporkan sebagai hasil utama skripsi ini,
sesuai catatan metodologis yang sama seperti yang diterapkan pada proyek sebelumnya di
repository ini.
