# Klasifikasi Kanker Paru-Paru pada Citra CT menggunakan EfficientNet-B0

Skripsi — Muhammad Ilhamsyah Ridwan (140810220059)
Program Studi Teknik Informatika, FMIPA, Universitas Padjadjaran

## Deskripsi

Penelitian ini mengembangkan model klasifikasi biner (cancer vs no_cancer)
pada citra CT paru-paru menggunakan arsitektur EfficientNet-B0 dengan
transfer learning, fine-tuning, dan data augmentation.

> **Catatan penting soal modalitas citra:** dataset sumber di Kaggle diberi
> nama "Lung Cancer **MRI** Images", tapi setelah diperiksa langsung, seluruh
> sampel citranya menunjukkan ciri khas CT scan (field-of-view berbentuk
> lingkaran dari rekonstruksi fan-beam, tulang rusuk bersinyal tinggi/putih,
> windowing paru klasik) — ciri yang tidak mungkin dihasilkan MRI. Penelitian
> ini mengoreksi label modalitas tersebut menjadi **CT** dan mendokumentasikan
> temuan ini di `notebooks/01_eda.ipynb`.

## Dataset

- Source utama: [Lung Cancer MRI Images](https://www.kaggle.com/datasets/xiaopengzhang12/lung-cancer-mri-images) (dilabel ulang sebagai citra CT, lihat catatan di atas)
- Total citra mentah: 3.680 (1.874 cancer, 1.806 no_cancer)
- Setelah deduplikasi (perceptual hash) + pembuangan 11 grup citra dengan label
  bertentangan: **2.875 citra unik** (rasio kelas ~42:58, cancer:no_cancer)
- Split internal: 70% train / 15% val / 15% test (stratified, dari data yang sudah bersih)
- Dataset tambahan: [IQ-OTH/NCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset)
  (1.054 citra CT unik setelah dedup, RS & scanner berbeda) — digabungkan ke
  training set (bukan ke val/test) untuk menambah keragaman data secara sah

## Hasil Model

| Konfigurasi | Test accuracy | ROC-AUC | Recall cancer |
|---|---|---|---|
| v6 (single model) | 72.5% | 0.782 | 68.5% |
| **Ensemble v5+v6 (dipakai di inferensi/`src/predict.py`)** | **73.2%** | **0.786** | **67.4%** |
| Ensemble 5-fold CV (dicoba, tidak dipakai) | 75.7% | 0.788 | 59.1% |

Ensemble 5-fold CV (`src/kfold_train.py`) memberi akurasi & AUC tertinggi, tapi recall
cancer turun cukup jauh (model jadi lebih konservatif). Karena false negative pada kanker
lebih berisiko secara klinis daripada false positive, **ensemble v5+v6 tetap dipilih
sebagai model final** meski akurasi agregatnya sedikit lebih rendah — lihat
`notebooks/03_train.ipynb` Bagian 9 untuk detail lengkap.

Lihat `reports/ablation_comparison.json`, `reports/ensemble_comparison.json`,
dan `notebooks/03_train.ipynb` untuk perbandingan lengkap **7 versi
eksperimen training** — termasuk versi awal yang performanya lebih tinggi
tapi ternyata terinflasi oleh data leakage antar split, dan sebuah insiden
leakage kedua (holdout eksternal 98.6% akurasi palsu akibat slice CT dari
pasien yang sama tersebar ke train/holdout) yang ditemukan dan diperbaiki
selama penelitian.

**Kenapa bukan ~90%?** Tujuh konfigurasi berbeda (regularisasi, arsitektur,
ukuran data) semuanya konvergen ke ROC-AUC ~0.77-0.79 setelah leakage
dibersihkan, dan preprocessing crop latar (`src/investigate_crop_preprocessing.py`)
terbukti berdampak minimal (rata-rata cuma 2.8% pengurangan area — citra
di dataset ini sudah dipotong cukup rapat). Ini menunjukkan batasnya ada di
ukuran/kualitas data (klasifikasi 1 slice CT 2D, bukan volume 3D), bukan lagi
di pilihan model. Detail lengkap di `notebooks/03_train.ipynb` Bagian 8.

## Inferensi: deteksi out-of-distribution + ensemble

`src/predict.py` membungkus pipeline inferensi lengkap:

1. **Deteksi OOD** — citra input diekstrak jadi fitur (embedding 1280-dim dari
   model v6), lalu diukur jaraknya (k-NN) ke seluruh embedding training. Kalau
   terlalu jauh dari distribusi citra CT paru (threshold dikalibrasi dari
   persentil ke-99 skor validation set, lihat `src/build_ood_detector.py`),
   citra ditolak dengan pesan "bukan citra CT paru-paru" — model tidak
   memaksakan jawaban cancer/no_cancer pada input yang tidak relevan.
2. **Klasifikasi** — kalau lolos, prediksi dihitung dari ensemble rata-rata
   softmax model v5+v6 (bukan test-time augmentation flip, yang terbukti
   menurunkan performa karena CT paru punya orientasi kiri-kanan yang
   bermakna — lihat `src/eval_ensemble.py`).

```bash
python src/predict.py path/ke/citra.png
```

## Tech Stack

- Python 3.11
- PyTorch 2.5 + CUDA 12.1
- timm (EfficientNet-B0)
- Streamlit (demo app)

## Struktur Folder

```
skripsi/
├── dataset/            # Raw dataset (gitignored)
├── dataset_split/       # Stratified split 70/15/15, hasil dedup (gitignored)
├── notebooks/           # Jupyter notebooks (EDA, split, training)
├── src/                 # Reusable Python modules (dataset, model, engine, train)
├── models/              # Trained model checkpoints (gitignored)
├── reports/             # Plots, confusion matrix, ablation study, dll
└── streamlit_app/       # Demo app
```

## Setup

```bash
uv venv --python 3.11
.venv\Scripts\activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
```

## Demo App

`streamlit_app/app.py` — antarmuka web untuk mengunggah citra CT dan melihat hasil
klasifikasi (dengan deteksi OOD bawaan). Dijalankan dengan:

```bash
streamlit run streamlit_app/app.py
```

> Ditujukan sebagai demo hasil skripsi, **bukan** perangkat lunak medis yang
> tersertifikasi — aplikasi menampilkan disclaimer ini secara eksplisit ke pengguna.

## License

MIT
