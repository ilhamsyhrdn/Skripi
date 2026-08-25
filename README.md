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

## Hasil Model (v6, final)

| Metrik | Nilai |
|---|---|
| Model | EfficientNet-B0, 224px |
| Train / Val / Test | 2.855 / 431 / 432 |
| Test accuracy | 72.5% |
| ROC-AUC | 0.782 |
| Recall kelas cancer | 68.5% |
| Train-test gap | 16.3% |

Lihat `reports/ablation_comparison.json` dan `notebooks/03_train.ipynb` untuk
perbandingan lengkap **7 versi eksperimen** — termasuk versi awal yang
performanya lebih tinggi tapi ternyata terinflasi oleh data leakage antar
split, dan sebuah insiden leakage kedua (holdout eksternal 98.6% akurasi palsu
akibat slice CT dari pasien yang sama tersebar ke train/holdout) yang
ditemukan dan diperbaiki selama penelitian.

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
