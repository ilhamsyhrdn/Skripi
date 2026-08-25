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
| v6 (single model, final individual model) | 72.5% | 0.782 | 68.5% |
| **Ensemble v5+v6 (dipakai di inferensi/`src/predict.py`)** | **73.2%** | **0.786** | 67.4% |

Lihat `reports/ablation_comparison.json`, `reports/ensemble_comparison.json`,
dan `notebooks/03_train.ipynb` untuk perbandingan lengkap **7 versi
eksperimen training** — termasuk versi awal yang performanya lebih tinggi
tapi ternyata terinflasi oleh data leakage antar split, dan sebuah insiden
leakage kedua (holdout eksternal 98.6% akurasi palsu akibat slice CT dari
pasien yang sama tersebar ke train/holdout) yang ditemukan dan diperbaiki
selama penelitian.

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

## License

MIT
