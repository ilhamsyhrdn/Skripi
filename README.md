# Klasifikasi Kanker Paru-Paru pada Citra MRI menggunakan EfficientNet-B0

Skripsi — Muhammad Ilhamsyah Ridwan (140810220059)
Program Studi Teknik Informatika, FMIPA, Universitas Padjadjaran

## Deskripsi

Penelitian ini mengembangkan model klasifikasi biner (cancer vs no_cancer)
pada citra MRI paru-paru menggunakan arsitektur EfficientNet-B0 dengan
transfer learning, fine-tuning, dan data augmentation.

## Dataset

- Source: [Lung Cancer MRI Images](https://www.kaggle.com/datasets/xiaopengzhang12/lung-cancer-mri-images)
- Total: 3.680 citra (1.874 cancer, 1.806 no_cancer)
- Format: PNG (3.668) + JPG (12)
- Split: 70% train / 15% val / 15% test (stratified)

## Tech Stack

- Python 3.11
- PyTorch 2.5 + CUDA 12.1
- timm (EfficientNet-B0)
- Streamlit (demo app)

## Struktur Folder

```
skripsi/
├── dataset/            # Raw dataset (gitignored)
├── dataset_split/       # Stratified split 70/15/15 (gitignored)
├── notebooks/           # Jupyter notebooks (EDA, training)
├── src/                 # Reusable Python modules
├── models/              # Trained model checkpoints (gitignored)
├── reports/             # Plots, confusion matrix, dll
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
