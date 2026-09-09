# Kodingan — Klasifikasi Kanker Paru-Paru CT (EfficientNet-B0 + Fine-Tuning + Augmentation + Ensemble)

Kode pendukung skripsi *"Optimasi Transfer Learning EfficientNet-B0 untuk Klasifikasi Kanker
Paru-Paru pada Citra Computed Tomography (CT) Menggunakan Fine-Tuning, Data Augmentation, dan
Ensemble Model"*.

## Struktur folder

```
Kodingan/
├── .venv/                     virtual environment (PyTorch + CUDA 12.1, dibuat lokal)
├── src/
│   ├── audit/                 audit dataset: deduplikasi, deteksi leakage, split, naive-baseline
│   ├── data/                  PyTorch Dataset + transform/augmentation
│   ├── models/                model factory (EfficientNet-B0, ResNet50) + freeze/unfreeze
│   ├── training/               engine (train/eval loop) + skrip 5-fold CV training
│   ├── evaluation/             evaluasi ensemble pada held-out test set
│   └── inference/               predictor (ensemble soft-voting) + Grad-CAM manual
├── streamlit_app/app.py         dashboard demo (prediksi, perbandingan model, audit dataset)
├── outputs/
│   ├── manifests/                 manifest CSV/JSON hasil audit & split
│   ├── models/                     checkpoint 10 model (*.pt)
│   ├── reports/                    history training + hasil evaluasi ensemble (JSON)
│   └── logs/                       log training & audit (live progress, bisa di-tail)
└── requirements.txt
```

## Menjalankan ulang seluruh pipeline dari awal

Dari dalam folder `Kodingan/` (PowerShell atau Git Bash):

```bash
# 1) audit dataset mentah -> manifest kanonik yang bersih dari duplikat/leakage
.venv/Scripts/python.exe -m src.audit.build_manifest
.venv/Scripts/python.exe -m src.audit.make_splits

# 2) (opsional, untuk bukti dampak leakage) baseline naif yang sengaja bocor
.venv/Scripts/python.exe -m src.audit.naive_leaky_baseline

# 3) training 5-fold cross-validation, 2 arsitektur (progress epoch tercetak live)
.venv/Scripts/python.exe -m src.training.train_cv --arch efficientnet_b0
.venv/Scripts/python.exe -m src.training.train_cv --arch resnet50

# 4) evaluasi ensemble pada held-out test set
.venv/Scripts/python.exe -m src.evaluation.ensemble_eval

# 5) jalankan dashboard
.venv/Scripts/python.exe -m streamlit run streamlit_app/app.py
```

## Ringkasan hasil (lihat `outputs/reports/ensemble_evaluation.json` untuk angka lengkap)

Dataset mentah: 12.882 citra dari 7 dataset Kaggle → setelah audit deduplikasi & anti-leakage
(lihat `D:\skripsi\Kejanggalan Dataset\`) → **894 citra unik** (Malignant 665 / Normal 136 /
Benign 93), displit menjadi held-out test (165 citra) + 5-fold CV (729 citra).

Ensemble akhir (10 model: 5× EfficientNet-B0 + 5× ResNet50, soft-voting) pada held-out test set:

| Metrik | Nilai |
|---|---:|
| Akurasi | ~85,5% |
| Macro-F1 | ~0,695 |
| Recall kelas Malignant ("cancer recall") | ~92,6% |
| Macro ROC-AUC | ~0,937 |

Alternatif ensemble EfficientNet-B0-only (5-fold) mencapai cancer recall tertinggi (~95,9%) dengan
akurasi sedikit lebih rendah (~84,2%) — kedua konfigurasi dilaporkan berdampingan di dashboard
karena keduanya valid tergantung prioritas klinis (maksimalkan deteksi kanker vs keseimbangan
antar-kelas). Lihat halaman "Perbandingan Model" pada dashboard Streamlit.

## Catatan penting

- Model & pipeline ini dilatih pada **PyTorch**, bukan TensorFlow/Keras seperti draf proposal awal,
  karena dukungan GPU native TensorFlow di Windows sudah dihentikan sejak versi 2.11 (perlu WSL2).
  PyTorch dipilih agar dapat memanfaatkan GPU NVIDIA RTX 4060 secara native di Windows. Seluruh
  konsep metodologi pada proposal (transfer learning, freeze/unfreeze bertahap, fine-tuning,
  data augmentation, evaluasi accuracy/precision/recall/F1) diimplementasikan identik, hanya
  frameworknya berbeda.
- Detail lengkap semua kejanggalan dataset yang ditemukan (duplikasi lintas dataset, data sintetis,
  data tanpa label, dsb.) ada di `D:\skripsi\Kejanggalan Dataset\`.
