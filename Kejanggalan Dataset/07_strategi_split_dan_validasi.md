# Strategi Split Train / Validation / Test yang Aman dari Kebocoran Data

Dokumen ini merangkum bagaimana temuan pada dokumen 01-06 diterjemahkan menjadi prosedur split data
yang dipakai untuk training (`Kodingan/src/audit/make_splits.py`).

## 1. Unit split bukan "gambar", tapi "grup"

Setiap baris pada `canonical_pool.csv` sudah merupakan representasi dari satu grup duplikat/near-
duplikat hasil audit (dokumen 01-03). Sebagai margin keamanan tambahan, setiap grup kemudian
dikelompokkan lagi menjadi `split_group = (sumber_dataset, folder_mentah, nomor_kasus // 5)` —
supaya nomor kasus yang berdekatan pada folder sumber yang sama (yang terbukti bisa jadi slice/crop
dari studi yang sama, lih. dokumen 03) tidak pernah terpisah antar split.

## 2. Test set diambil lebih dulu dan tidak pernah disentuh lagi

18% dari setiap kelas dialokasikan ke **held-out test set** melalui alokasi acak per `split_group`
(bukan per gambar) sampai target proporsi tercapai. Set ini disimpan di `test_holdout.csv` dan **baru
dipakai sekali di akhir** untuk melaporkan performa akhir ensemble — tidak pernah dipakai untuk
memilih hyperparameter, arsitektur, atau titik *early stopping* selama proses pengembangan.

## 3. Sisa data (82%) di-*cross-validation* 5-fold, bukan displit tunggal 70/15/15

Proposal skripsi (bab 3.5.1) awalnya merujuk pembagian train/val/test 70/15/15. Karena setelah
deduplikasi total data bersih hanya 894 gambar (dan kelas Benign hanya 93), sebuah split tunggal
70/15/15 akan menyisakan validation/test per kelas yang sangat kecil (Benign ±14 gambar) sehingga
metrik akan sangat tidak stabil (satu-dua kesalahan prediksi bisa mengubah recall puluhan persen).

Sebagai gantinya, digunakan **`StratifiedGroupKFold` 5-fold** (scikit-learn) pada 82% data
train+validation, yang secara bersamaan (a) menjaga proporsi kelas tetap seimbang di setiap fold, dan
(b) menjamin `split_group` yang sama tidak pernah muncul di lebih dari satu fold. Pendekatan ini juga
sejalan dengan metodologi Shi et al. (2022) dan Wang et al. (2022) pada tinjauan pustaka skripsi ini,
yang sama-sama memakai 5-fold cross-validation untuk klasifikasi nodul paru dari CT.

Kelima model yang dihasilkan dari 5-fold ini (per arsitektur) menjadi anggota **ensemble** —
memenuhi komponen "Ensemble Model" pada judul skripsi sekaligus menstabilkan performa pada dataset
kecil yang timpang (lih. `05_ketidakseimbangan_kelas.md`).

## 4. Verifikasi otomatis anti-kebocoran

Skrip `make_splits.py` menjalankan dua pemeriksaan (`assert`) sebelum menyimpan hasil split:

```python
fold_of_group = trainval_df.groupby("split_group")["fold"].nunique()
assert (fold_of_group == 1).all(), "A split_group leaked across folds!"
assert not (set(trainval_df["split_group"]) & test_groups), "A split_group leaked into test!"
```

Jika salah satu kondisi ini gagal, skrip akan berhenti dengan error alih-alih diam-diam menghasilkan
split yang bocor.

## Hasil split final

| Split | Malignant | Normal | Benign | Total |
|---|---:|---:|---:|---:|
| Test (held-out, 18%) | 122 | 26 | 17 | 165 |
| Train+val (82%, untuk 5-fold CV) | 543 | 110 | 76 | 729 |
| — rata-rata per fold (validation) | ≈109 | ≈22 | ≈15 | ≈146 |

Sumber angka: `Kodingan/outputs/manifests/split_summary.json`.
