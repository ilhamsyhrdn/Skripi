"""Renumber Bab IV and insert the sections that were missing: model
implementation with code listings, training-process analysis (curves +
per-epoch table), Grad-CAM discussion, and input-validation results.

Renumbering is done through placeholder tokens so that overlapping maps
(4.3 -> 4.4 while 4.4 -> 4.5) cannot collide.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path("D:/skripsi/Naskah Skripsi/BAB_IV_Hasil_dan_Pembahasan.md")

SECTION_MAP = {"4.6": "4.7", "4.7": "4.8", "4.8": "4.9", "4.9": "4.10",
               "4.10": "4.11", "4.11": "4.14"}
TABLE_MAP = {"4.3": "4.4", "4.4": "4.5", "4.5": "4.6", "4.6": "4.7",
             "4.7": "4.8", "4.8": "4.10"}
FIGURE_MAP = {"4.1": "4.3", "4.2": "4.4", "4.3": "4.5", "4.4": "4.6"}
FILE_MAP = {
    "Gambar_4.1_Confusion_Matrix_Stacking.png": "Gambar_4.3_Confusion_Matrix_Stacking.png",
    "Gambar_4.2_Kurva_ROC_Stacking.png": "Gambar_4.4_Kurva_ROC_Stacking.png",
    "Gambar_4.3_Sweep_Ambang_Keputusan.png": "Gambar_4.5_Sweep_Ambang_Keputusan.png",
    "Gambar_4.4_Perbandingan_Skenario.png": "Gambar_4.6_Perbandingan_Skenario.png",
}

IMPLEMENTASI = """
### 4.1.1 Pra-pemrosesan Data dan Augmentasi

Setiap citra masukan melewati rangkaian transformasi yang sama sebelum masuk ke model: pengubahan ukuran menjadi 224×224 piksel, konversi ke tensor, dan normalisasi memakai statistik ImageNet (*mean* [0,485; 0,456; 0,406] dan standar deviasi [0,229; 0,224; 0,225]). Normalisasi dengan statistik ImageNet dipilih karena bobot awal kedua arsitektur memang dilatih pada distribusi tersebut, sehingga masukan yang selaras mempercepat konvergensi pada tahap *fine-tuning*.

Perbedaan perlakuan hanya terletak pada data latih, yang diperkaya dengan augmentasi acak berupa pembalikan horizontal, transformasi afin (rotasi, translasi, dan skala), serta perubahan warna. Data validasi dan data uji sengaja tidak diaugmentasi agar angka evaluasi mencerminkan performa pada citra apa adanya.

*script kode program pra-pemrosesan dan augmentasi:*

```python
def build_transforms(image_size=224, train=False, augment_strength="medium"):
    base = [v2.ToImage(), v2.Resize((image_size, image_size), antialias=True)]
    if train:
        base += [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.92, 1.08)),
            v2.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    base += [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return v2.Compose(base)
```

### 4.1.2 Arsitektur dan Konfigurasi Model

Kedua arsitektur dimuat dengan bobot *pre-trained* ImageNet, lalu lapisan klasifikasi terakhirnya diganti dengan lapisan *fully connected* berukuran tiga keluaran sesuai jumlah kelas. Strategi pelatihannya dua fase: pada fase A seluruh *backbone* dibekukan dan hanya lapisan klasifikasi baru yang dilatih, kemudian pada fase B tiga blok terakhir *backbone* dibuka dan dilatih ulang dengan laju pembelajaran seratus kali lebih kecil.

*script kode program pemuatan model dan pembekuan backbone:*

```python
def build_model(arch: str, num_classes: int = 3):
    if arch == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone(model, arch):
    for p in model.parameters():
        p.requires_grad = False
    head = model.classifier if arch == "efficientnet_b0" else model.fc
    for p in head.parameters():
        p.requires_grad = True
```

Fungsi kerugian yang dipakai adalah *CrossEntropyLoss* dengan bobot per kelas berbanding terbalik terhadap frekuensi kemunculannya, sehingga kesalahan pada kelas Benign (kelas paling sedikit datanya) dihukum lebih berat daripada kesalahan pada kelas Malignant.

*script kode program bobot kelas dan konfigurasi dua fase:*

```python
def class_weights_from_df(df, device):
    counts = df["canonical_label"].value_counts()
    freqs = np.array([counts.get(c, 1) for c in CLASS_NAMES], dtype=np.float64)
    weights = freqs.sum() / (len(CLASS_NAMES) * freqs)
    return torch.tensor(weights, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights_from_df(train_df, device))

# Fase A: feature extraction
freeze_backbone(model, arch)
opt_a = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
fit(model, ..., epochs=12, phase_name="A-head", patience=6)

# Fase B: fine-tuning
unfreeze_for_finetune(model, arch, n_blocks=3)
opt_b = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
sched_b = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_b, mode="max",
                                                     factor=0.5, patience=3)
fit(model, ..., epochs=25, phase_name="B-finetune", scheduler=sched_b, patience=6)
```

### 4.1.3 Skema Pembagian Data dan Pengujian

Pembagian data memakai *StratifiedGroupKFold* dengan lima lipatan, dengan kolom `split_group` sebagai penanda grup (kunci kasus untuk data Kaggle, ID pasien untuk LIDC-IDRI). Sebelum pelatihan dijalankan, program memverifikasi bahwa tidak ada satu pun grup yang muncul di dua lipatan sekaligus maupun bocor ke data uji; bila verifikasi ini gagal, program berhenti dan menolak melanjutkan pelatihan.

*script kode program pembagian data dan pemeriksaan kebocoran:*

```python
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (_, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
    trainval.loc[trainval.index[val_idx], "fold"] = fold

# anti-leakage assertions: satu grup tidak boleh berada di dua fold
for group, sub in trainval.groupby("split_group"):
    assert sub["fold"].nunique() == 1, f"grup {group} tersebar di beberapa fold"
assert set(trainval["split_group"]) & set(test_df["split_group"]) == set()
```
"""

PELATIHAN = """
## 4.6 Analisis Proses Pelatihan

Selain angka akhir, dinamika pelatihan itu sendiri perlu ditinjau untuk memastikan model benar-benar belajar dan tidak sekadar menghafal data latih. Gambar 4.1 menampilkan kurva *loss* dan akurasi pada ResNet50 *fold* 1, yaitu model tunggal dengan performa terbaik pada data uji (akurasi 77,97%).

![Gambar 4.1 Kurva Loss dan Akurasi ResNet50 Fold 1](Gambar/Gambar_4.1_Kurva_ResNet_Fold1.png)

*Gambar 4.1 Kurva Loss dan Akurasi pada ResNet50 Fold 1*

Kurva ini memperlihatkan dua hal sekaligus. Pertama, transisi dari fase A ke fase B (garis putus-putus vertikal) memang memberi dorongan: *loss* validasi yang sempat mendatar di kisaran 0,74–0,78 selama fase *feature extraction* mulai turun konsisten setelah *fine-tuning* dimulai, hingga mencapai 0,59 pada epoch terakhir. Kedua, jarak antara kurva latih dan kurva validasi melebar seiring bertambahnya epoch pada fase B: akurasi latih naik sampai 86,59% sementara akurasi validasi berhenti di 76,56%. Selisih sekitar sepuluh poin persentase ini menandakan *overfitting* ringan, yaitu kondisi ketika model mulai menyesuaikan diri terlalu jauh pada data latih. Mekanisme *early stopping* berbasis macro-F1 validasi menahan pelatihan agar tidak berlanjut lebih jauh dari titik ini, dan bobot yang disimpan adalah bobot dengan macro-F1 validasi tertinggi, bukan bobot dari epoch terakhir.

Rincian perkembangan tiap epoch pada *fold* yang sama disajikan pada Tabel 4.3 (ditampilkan sebagian: epoch awal, epoch transisi antar fase, dan epoch terbaik).

Tabel 4.3 Detail Epoch pada ResNet50 Fold 1

| Epoch | Fase | Laju Pembelajaran | Loss Latih | Akurasi Latih (%) | Loss Validasi | Akurasi Validasi (%) | Macro-F1 Validasi |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | A | 1e-03 | 0,9126 | 62,71 | 0,7808 | 69,71 | 0,6157 |
| 2 | A | 1e-03 | 0,7663 | 70,59 | 0,7852 | 69,29 | 0,6194 |
| 3 | A | 1e-03 | 0,7187 | 72,45 | 0,7871 | 68,67 | 0,6351 |
| 6 | A | 1e-03 | 0,6655 | 72,45 | 0,7573 | 68,26 | 0,6315 |
| 12 | A | 1e-03 | 0,6133 | 74,73 | 0,7435 | 70,12 | 0,6585 |
| 13 | B | 1e-05 | 0,6038 | 74,00 | 0,7673 | 68,67 | 0,6543 |
| 14 | B | 1e-05 | 0,5779 | 75,30 | 0,8023 | 68,05 | 0,6473 |
| 36 | B | 1e-05 | 0,3835 | 82,96 | 0,6348 | 75,10 | 0,7084 |
| **37** | **B** | **1e-05** | **0,3535** | **86,59** | **0,5938** | **76,56** | **0,7150** |

Terlihat bahwa satu epoch pertama setelah pergantian fase (epoch 13–14) justru sempat memperburuk *loss* validasi sebelum akhirnya membaik. Hal ini wajar karena pembukaan tiga blok terakhir *backbone* mengubah banyak parameter sekaligus, sehingga model memerlukan beberapa epoch untuk menyesuaikan diri pada laju pembelajaran yang baru.

Sebagai pembanding, Gambar 4.2 menampilkan kurva pada EfficientNet-B0 *fold* 0. Pola umumnya serupa, tetapi pelatihannya berhenti jauh lebih awal (22 epoch berbanding 37) karena *early stopping* terpicu lebih cepat, yang menjelaskan mengapa rata-rata performa EfficientNet-B0 sedikit di bawah ResNet50 pada Tabel 4.2.

![Gambar 4.2 Kurva Loss dan Akurasi EfficientNet-B0 Fold 0](Gambar/Gambar_4.2_Kurva_EffNet_Fold0.png)

*Gambar 4.2 Kurva Loss dan Akurasi pada EfficientNet-B0 Fold 0*
"""

GRADCAM = """
## 4.12 Visualisasi Grad-CAM

Untuk memeriksa apakah model memang memusatkan perhatian pada area yang relevan secara klinis, bukan pada artefak citra, dilakukan visualisasi Grad-CAM pada beberapa citra uji memakai ResNet50 *fold* 1. Peta panas dihitung dari gradien skor kelas terhadap *feature map* lapisan konvolusi terakhir, kemudian ditumpangkan pada citra aslinya. Hasilnya disajikan pada Gambar 4.7.

![Gambar 4.7 Visualisasi Grad-CAM](Gambar/Gambar_4.7_GradCAM.png)

*Gambar 4.7 Visualisasi Grad-CAM pada Citra Uji untuk Ketiga Kelas*

Hasil visualisasi ini memberikan gambaran yang berbeda-beda antar kelas, dan perbedaannya justru informatif:

1. **Kelas Malignant.** Peta panas terkonsentrasi pada area massa atau opasitas di dalam lapang paru, yaitu tepat pada struktur yang secara klinis memang menjadi dasar kecurigaan keganasan. Ini menunjukkan bahwa untuk kelas dengan performa terbaik (*recall* 91,5%), model mendasarkan keputusannya pada fitur yang masuk akal secara anatomis.

2. **Kelas Benign.** Peta panas justru menyala kuat di area luar lapang paru, yakni pada dinding dada dan bahu bagian atas. Temuan ini mengindikasikan model tidak menemukan fitur pembeda yang kuat di dalam paru untuk kelas ini, lalu bersandar pada karakteristik non-anatomis, kemungkinan berupa perbedaan pembingkaian atau kontras yang kebetulan berkorelasi dengan sumber data kelas Benign. Temuan ini sejalan dengan rendahnya *recall* kelas Benign (30,0%) dan memberi penjelasan yang lebih dalam atas angka tersebut: masalahnya bukan semata-mata jumlah data yang sedikit, melainkan model belum mempelajari fitur intrinsik nodul jinak.

3. **Kelas Normal.** Perhatian model terpusat di area mediastinum dan jantung, bukan pada parenkim paru. Interpretasi yang masuk akal adalah model mengenali kelas Normal dari ketiadaan anomali di lapang paru, sehingga aktivasinya jatuh pada struktur besar yang paling konsisten muncul di semua citra sehat.

Dengan demikian, Grad-CAM tidak sekadar berfungsi sebagai fitur pelengkap aplikasi, tetapi juga menjadi alat diagnosis terhadap model itu sendiri. Untuk kelas Malignant, visualisasi ini memperkuat kepercayaan terhadap keputusan model; untuk kelas Benign, visualisasi ini justru mengungkap keterbatasan yang tidak terlihat dari angka akurasi semata, dan menjadi dasar saran perbaikan pada Bab V.

## 4.13 Hasil Validasi Input (Deteksi Citra di Luar Domain)

Lapisan validasi input diuji secara terpisah dari model klasifikasi utama. Model klasifikasi biner EfficientNet-B0 dilatih untuk membedakan citra CT paru-paru dari citra objek umum (COCO), lalu dievaluasi pada 376 citra uji yang terbagi seimbang: 188 citra CT paru-paru dan 188 citra bukan CT. Hasilnya ditampilkan pada Gambar 4.8 dan Tabel 4.9.

![Gambar 4.8 Confusion Matrix Validasi Input](Gambar/Gambar_4.8_Confusion_Validasi_Input.png)

*Gambar 4.8 Confusion Matrix Model Validasi Input*

Tabel 4.9 Hasil Evaluasi Model Validasi Input

| Kelas | Support | Presisi | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Bukan Citra CT Paru-paru | 188 | 1,000 | 1,000 | 1,000 |
| Citra CT Paru-paru | 188 | 1,000 | 1,000 | 1,000 |
| **Akurasi keseluruhan** | **376** | | | **100%** |

Model validasi input mencapai akurasi sempurna pada data uji, tanpa satu pun kesalahan klasifikasi. Hasil ini perlu dibaca dengan proporsional: tugas membedakan citra CT paru-paru dari foto objek sehari-hari jauh lebih mudah daripada membedakan nodul ganas dari nodul jinak, karena kedua kelasnya berbeda secara mencolok pada hampir setiap aspek visual (skala keabuan versus warna, struktur anatomi versus objek bebas). Akurasi 100% pada tugas ini karena itu bukan indikasi bahwa keseluruhan sistem sempurna, melainkan bahwa lapisan penyaring bekerja andal untuk fungsi spesifiknya, yaitu mencegah citra yang sama sekali tidak relevan diproses oleh model klasifikasi utama.

Satu keterbatasan perlu dicatat secara terbuka: model validasi input ini dilatih memakai pool citra LIDC-IDRI hasil *crop*, bukan pool gabungan final. Pengujian di atas juga hanya mencakup pembeda ekstrem (CT paru-paru versus objek umum), belum menguji kasus yang lebih menantang seperti citra CT organ lain atau citra MRI, yang secara visual jauh lebih mirip dengan CT paru-paru. Pengujian terhadap kasus batas semacam itu menjadi salah satu saran pengembangan pada Bab V.
"""


def renumber(text, mapping, pattern_tpl):
    """Two-phase replace via placeholders to avoid chained collisions."""
    tmp = text
    for old, new in mapping.items():
        tmp = re.sub(pattern_tpl.format(num=re.escape(old)),
                     lambda m, n=new: m.group(0).replace(old, f"@@{n}@@"), tmp)
    return tmp.replace("@@", "")


def main():
    t = SRC.read_text(encoding="utf-8")

    # 1) section headings and in-text "Bagian 4.x" references
    t = renumber(t, SECTION_MAP, r"## {num} ")
    # 2) tables (captions and cross-references)
    t = renumber(t, TABLE_MAP, r"Tabel {num}\b")
    # 3) figures (captions, cross-references, alt text)
    t = renumber(t, FIGURE_MAP, r"Gambar {num}\b")
    # 4) image file names
    for old, new in FILE_MAP.items():
        t = t.replace(old, new)

    # 5) insert implementation subsections into 4.1
    anchor = ("Seluruh model pada tiap skenario dilatih dengan arsitektur, "
              "*hyperparameter*, dan skema *Stratified Group K-Fold* yang identik "
              "sebagaimana dijabarkan pada Bab III, agar perbedaan hasil antar "
              "skenario dapat diatribusikan pada perbedaan data, bukan perbedaan "
              "konfigurasi pelatihan.")
    assert anchor in t, "anchor 4.1 tidak ditemukan"
    t = t.replace(anchor, anchor + "\n" + IMPLEMENTASI.rstrip() + "\n")

    # 6) insert training-analysis section before the (renumbered) 4.7
    marker = "## 4.7 Skenario E dan F"
    assert marker in t, "penanda 4.7 tidak ditemukan"
    t = t.replace(marker, PELATIHAN.strip() + "\n\n" + marker)

    # 7) insert Grad-CAM + input validation before the (renumbered) 4.14
    marker = "## 4.14 Evaluasi Black Box Testing"
    assert marker in t, "penanda 4.14 tidak ditemukan"
    t = t.replace(marker, GRADCAM.strip() + "\n\n" + marker)

    SRC.write_text(t, encoding="utf-8")

    caps_t = re.findall(r"^Tabel (\d+\.\d+)", t, flags=re.M)
    caps_g = re.findall(r"^\*Gambar (\d+\.\d+)", t, flags=re.M)
    print("urutan caption Tabel :", caps_t)
    print("urutan caption Gambar:", caps_g)


if __name__ == "__main__":
    main()
