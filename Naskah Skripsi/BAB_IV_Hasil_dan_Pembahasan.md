# BAB IV

# HASIL DAN PEMBAHASAN

Bab ini memaparkan hasil penelitian dan disusun mengikuti tiga teknik optimasi yang dinyatakan pada judul: *fine-tuning*, *data augmentation*, dan *ensemble model*. Setelah lingkungan implementasi dijabarkan, kontribusi tiap teknik diukur satu per satu lewat perbandingan terkendali, yaitu dengan mengubah hanya satu faktor dan mempertahankan sisanya. Dengan cara ini, kenaikan performa dapat diatribusikan pada teknik tertentu, bukan sekadar dilaporkan sebagai satu angka akhir yang tidak diketahui asal usulnya. Hasil pengukuran tersebut dilengkapi analisis kesalahan klasifikasi dan pemecahan performa menurut sumber data. Rangkaian analisis ini kemudian bermuara pada tahap implementasi, yaitu ketika konfigurasi final beserta model validasi input diintegrasikan ke dalam prototipe aplikasi berbasis *web*, lalu diuji fungsionalitasnya memakai *Black Box Testing*.

## 4.1 Lingkungan dan Implementasi Model

Seluruh model dilatih dengan arsitektur, *hyperparameter*, dan skema *Stratified Group K-Fold* yang identik sebagaimana dijabarkan pada Bab III. Konsistensi ini penting karena keseluruhan Bab IV bersandar pada perbandingan antar konfigurasi: bila konfigurasi pelatihannya ikut berubah, perbedaan hasil tidak lagi dapat diatribusikan pada teknik yang sedang diuji.

### 4.1.1 Pra-pemrosesan Data dan Augmentasi

Setiap citra masukan melewati rangkaian transformasi yang sama sebelum masuk ke model: pengubahan ukuran menjadi 512×512 piksel, konversi ke *tensor*, dan normalisasi memakai statistik ImageNet (*mean* [0,485; 0,456; 0,406] dan standar deviasi [0,229; 0,224; 0,225]). Normalisasi dengan statistik ImageNet dipilih karena bobot awal kedua arsitektur memang dilatih pada distribusi tersebut, sehingga masukan yang selaras mempercepat konvergensi pada tahap *fine-tuning*.

Resolusi 512×512 dipakai karena citra CT pada kedua sumber memang tersimpan pada ukuran asli 512×512 piksel. Menurunkannya ke 224×224, sebagaimana lazim dilakukan pada pelatihan ImageNet, berarti membuang lebih dari tiga perempat piksel yang tersedia. Pada citra irisan dada utuh, nodul hanya menempati sebagian kecil luas citra, sehingga informasi yang hilang akibat penurunan resolusi justru mengenai bagian yang paling menentukan. Resolusi tidak dinaikkan melewati 512 karena di atas titik itu piksel tambahan hanya hasil interpolasi, bukan informasi baru dari alat pemindai.

Perbedaan perlakuan hanya terletak pada data latih, yang diperkaya dengan augmentasi acak. Kode di bawah memuat empat preset kekuatan augmentasi, tetapi penelitian ini hanya memakai dua di antaranya. Preset `ct` menjadi konfigurasi final, yaitu pembalikan horizontal dan rotasi acak hingga 7° saja, tanpa perubahan kecerahan maupun kontras, tanpa translasi, dan tanpa penskalaan. Preset `medium` dipakai sebagai konfigurasi augmentasi penuh pada eksperimen pembanding di Bagian 4.4, sedangkan preset `light` dan `heavy` tidak dipakai. Konfigurasi tanpa augmentasi pada eksperimen yang sama dijalankan lewat opsi terpisah, yaitu dengan memberikan transformasi data validasi (`train=False`) kepada data latih, sehingga seluruh transformasi acak terlewati. Data validasi dan data uji sengaja tidak diaugmentasi agar angka evaluasi mencerminkan performa pada citra apa adanya.

*script* kode program pra-pemrosesan dan augmentasi:
```python
def build_transforms(image_size: int = 224, train: bool = True, augment_strength: str = "medium") -> v2.Compose:
    # ... (cabang tanpa augmentasi untuk data validasi dan uji, lihat Bagian 4.9.3) ...

    # "ct" keeps only the variation a chest CT can actually show. Hounsfield
    # units are physically calibrated, so a grey level means a tissue density:
    # jittering brightness/contrast corrupts signal rather than nuisance. And a
    # nodule spans a few pixels of a 512px slice, so translation and rescaling
    # can resample it away. Flip and a small rotation survive both objections.
    strengths = {
        "ct": dict(rot=7, jitter=0.0, translate=0.0, scale=(1.0, 1.0)),
        "light": dict(rot=10, jitter=0.1, translate=0.05, scale=(0.95, 1.05)),
        "medium": dict(rot=15, jitter=0.2, translate=0.1, scale=(0.9, 1.1)),
        "heavy": dict(rot=20, jitter=0.3, translate=0.15, scale=(0.85, 1.15)),
    }
    p = strengths[augment_strength]
    steps = [
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
    ]
    if p["translate"] or p["scale"] != (1.0, 1.0):
        steps.append(v2.RandomAffine(degrees=p["rot"],
                                     translate=(p["translate"], p["translate"]),
                                     scale=p["scale"]))
    elif p["rot"]:
        steps.append(v2.RandomRotation(degrees=p["rot"]))
    if p["jitter"]:
        steps.append(v2.ColorJitter(brightness=p["jitter"], contrast=p["jitter"]))
    steps += [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return v2.Compose(steps)
```

Pembalikan vertikal sengaja tidak dipakai sama sekali, sebab citra CT dada yang terbalik atas-bawah bukan variasi yang wajar dan justru akan mengajari model pola yang keliru. Adapun kekuatan augmentasi mana yang paling sesuai untuk citra CT merupakan pertanyaan tersendiri yang diuji secara langsung pada Bagian 4.4, bukan diasumsikan sejak awal.

### 4.1.2 Arsitektur dan Konfigurasi Model

Kedua arsitektur dimuat dengan bobot *pre-trained* ImageNet, lalu lapisan klasifikasi terakhirnya diganti dengan rangkaian *Dropout* berprobabilitas 0,3 diikuti lapisan *fully connected* berukuran tiga keluaran sesuai jumlah kelas. *Dropout* dipasang tepat sebelum lapisan keputusan sebagai peredam *overfitting* pada bagian jaringan yang dilatih dari nol, yaitu bagian yang paling rawan menghafal karena bobot awalnya acak dan datanya terbatas. Strategi pelatihannya dua fase: pada fase A seluruh *backbone* dibekukan dan hanya lapisan klasifikasi baru yang dilatih, kemudian pada fase B tiga blok terakhir *backbone* dibuka dan dilatih ulang dengan laju pembelajaran seratus kali lebih kecil.

*script* kode program pemuatan model dan pembekuan *backbone*:
```python
def build_model(arch: str = "efficientnet_b0", n_classes: int = 3, dropout: float = 0.3):
    if arch == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, n_classes),
        )
        return model
    if arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, n_classes))
        return model
    raise ValueError(f"unknown arch: {arch}")


def freeze_backbone(model, arch):
    for p in model.parameters():
        p.requires_grad = False
    head = model.classifier if arch == "efficientnet_b0" else model.fc
    for p in head.parameters():
        p.requires_grad = True
```

Fungsi kerugian yang dipakai adalah *CrossEntropyLoss* dengan bobot per kelas berbanding terbalik terhadap frekuensi kemunculannya, sehingga kesalahan pada kelas Benign (kelas paling sedikit datanya) dihukum lebih berat daripada kesalahan pada kelas Malignant.

*script* kode program bobot kelas dan konfigurasi dua fase:
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

Pembagian data berlangsung dua tahap, keduanya memakai *StratifiedGroupKFold* dengan kolom `split_group` sebagai penanda grup (kunci kasus untuk data Kaggle, ID pasien untuk LIDC-IDRI) dan *random seed* 42. Tahap pertama memisahkan 18% grup sebagai *held-out test set*, dan citra berlabel perkiraan sengaja dikeluarkan dari undian ini sehingga hanya citra berlabel terpercaya yang berpeluang menjadi data uji. Tahap kedua membagi sisanya menjadi lima lipatan untuk validasi silang. Sebelum pelatihan dijalankan, program memverifikasi bahwa tidak ada grup yang muncul di dua lipatan sekaligus, tidak ada grup yang bocor ke data uji, dan tidak ada label perkiraan yang masuk data uji; bila salah satu pemeriksaan gagal, program berhenti dan menolak melanjutkan.

*script* kode program pembagian data dan pemeriksaan kebocoran:
```python
eligible = pool[pool["label_origin"] != "pseudo-knn"].reset_index(drop=True)
forced_train = pool[pool["label_origin"] == "pseudo-knn"].reset_index(drop=True)

n_test_folds = max(1, round(1 / TEST_FRACTION))
sgkf = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True, random_state=SEED)
train_idx, test_idx = next(sgkf.split(eligible, eligible["canonical_label"],
                                      groups=eligible["split_group"]))
test_df = eligible.iloc[test_idx].reset_index(drop=True)
trainval = eligible.iloc[train_idx].reset_index(drop=True)
trainval = pd.concat([trainval, forced_train], ignore_index=True)

sgkf2 = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
trainval["fold"] = -1
for fold, (_, val_idx) in enumerate(sgkf2.split(trainval, trainval["canonical_label"],
                                                groups=trainval["split_group"])):
    trainval.loc[trainval.index[val_idx], "fold"] = fold

# anti-leakage checks
assert (trainval["fold"] >= 0).all(), "ada baris tanpa fold"
for group, sub in trainval.groupby("split_group"):
    assert sub["fold"].nunique() == 1, f"grup {group} tersebar di beberapa fold"
overlap = set(trainval["split_group"]) & set(test_df["split_group"])
assert not overlap, f"grup bocor ke test: {list(overlap)[:3]}"
assert (test_df["label_origin"] != "pseudo-knn").all(), "label tebakan bocor ke data uji"
```

Komposisi akhir data yang dipakai seluruh eksperimen pada bab ini disajikan pada Tabel 4.1.

Tabel 4.1 Komposisi Data Pelatihan dan Pengujian

| Bagian | Malignant | Benign | Normal | Total |
|---|---:|---:|---:|---:|
| Data latih dan validasi (5 *fold*) | 1.655 | 315 | 399 | 2.369 |
| *Held-out test set* | 300 | 64 | 78 | 442 |
| **Total** | **1.955** | **379** | **477** | **2.811** |

Kelima *fold* berukuran hampir sama besar, yaitu 473 citra pada *fold* 0 dan 474 citra pada keempat *fold* lainnya. Seluruh angka evaluasi pada bab ini dihitung pada 442 citra *held-out test set*, yaitu data yang tidak pernah dilibatkan dalam pelatihan maupun pemilihan model manapun.

## 4.2 Hasil Dasar *Transfer Learning* EfficientNet-B0 dan ResNet50

Sebelum kontribusi masing-masing teknik optimasi diukur, performa dasar kesepuluh model tunggal perlu ditetapkan lebih dulu sebagai titik acuan. Setiap model dilatih pada empat *fold* dan divalidasi pada satu *fold* sisanya, lalu bobot pada *epoch* dengan macro-F1 validasi tertinggi diuji pada 442 citra *held-out test set*. Rekapitulasi hasilnya, yang dihitung berdasarkan rumus akurasi, macro-F1, *recall*, dan ROC-AUC pada Bab II, disajikan pada Tabel 4.2.

Tabel 4.2 Performa Model Tunggal per *Fold* pada Data Uji

| Model | *Epoch* Terbaik | Akurasi | Macro-F1 | *Cancer Recall* | *Benign Recall* | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 *fold* 0 | 10 | 67,42% | 0,648 | 62,67% | 76,56% | 0,881 |
| EfficientNet-B0 *fold* 1 | 23 | 69,68% | 0,662 | 65,00% | 73,44% | 0,893 |
| EfficientNet-B0 *fold* 2 | 15 | 66,52% | 0,645 | 60,00% | 78,12% | 0,891 |
| EfficientNet-B0 *fold* 3 | 34 | 68,33% | 0,656 | 64,67% | 76,56% | 0,901 |
| EfficientNet-B0 *fold* 4 | 18 | 66,29% | 0,630 | 62,00% | 70,31% | 0,890 |
| ResNet50 *fold* 0 | 25 | 72,40% | 0,670 | 74,33% | 67,19% | 0,904 |
| ResNet50 *fold* 1 | 31 | 73,53% | 0,698 | 71,67% | 78,12% | 0,904 |
| ResNet50 *fold* 2 | 14 | 69,00% | 0,644 | 69,00% | 73,44% | 0,880 |
| ResNet50 *fold* 3 | 24 | 71,27% | 0,669 | 69,33% | 73,44% | 0,903 |
| **ResNet50 *fold* 4** | **35** | **76,47%** | **0,710** | **77,67%** | 68,75% | **0,914** |
| **Rata-rata** | **-** | **70,09%** | **0,663** | **67,63%** | **73,59%** | **0,896** |

Kolom *epoch* terbaik memperlihatkan bahwa sembilan dari sepuluh model mencapai performa validasi tertingginya pada fase B, yaitu setelah *epoch* ke-12. Satu-satunya pengecualian adalah EfficientNet-B0 *fold* 0, yang bobot terbaiknya justru diperoleh pada *epoch* ke-10 di fase A. EfficientNet-B0 *fold* 3 mencatat *epoch* terbaik pada *epoch* terakhirnya, yaitu *epoch* ke-34, karena fase A pada *fold* ini berhenti lebih awal di *epoch* ke-9 lalu fase B berjalan penuh hingga batas 25 *epoch* dengan macro-F1 yang masih terus membaik.

Dua pola terbaca dari tabel ini. Pertama, ResNet50 secara konsisten mengungguli EfficientNet-B0 pada data ini, dengan rata-rata akurasi 72,53% berbanding 67,65%, yaitu selisih 4,89 poin persentase. Kedua, sebaran antar *fold* cukup lebar, dari 66,29% hingga 76,47%, yang berarti performa satu model tunggal ikut bergantung pada pembagian data yang kebetulan diperolehnya. Keragaman ini sendiri justru menjadi bahan baku yang berguna bagi teknik *ensemble* pada Bagian 4.5.

Perbandingan akurasi di atas tidak sepenuhnya mencerminkan kualitas kedua arsitektur pada seluruh kelas. EfficientNet-B0 justru lebih unggul pada kelas Benign, yaitu kelas tersulit dalam penelitian ini, dengan rata-rata *benign recall* 75,00% berbanding 72,19% milik ResNet50. Keunggulan ResNet50 terutama terletak pada kelas Malignant yang jumlah citranya paling banyak, sehingga pengaruhnya terhadap akurasi keseluruhan menjadi besar. Perbedaan watak inilah yang kemudian dimanfaatkan *meta-learner* pada Bagian 4.5.2.

Dinamika pelatihannya ditampilkan pada Gambar 4.1, memakai ResNet50 *fold* 0 sebagai contoh karena model tersebut mencapai macro-F1 validasi tertinggi di antara kesepuluh model.

![Gambar 4.1 Kurva Loss dan Akurasi ResNet50 Fold 0](Gambar/Gambar_4.1_Kurva_Pelatihan.png)

Gambar 4.1 Kurva *Loss* dan Akurasi pada ResNet50 *Fold* 0

Kurva latih dan kurva validasi bergerak searah sepanjang pelatihan. Pada *epoch* terbaiknya, akurasi latih mencapai 86,08% sementara akurasi validasi 80,76%, sehingga jurang di antara keduanya 5,31 poin persentase. Jurang sebesar itu menandakan *overfitting* yang masih terkendali, bukan model yang sekadar menghafal data latihnya. Pengaruh augmentasi terhadap besarnya jurang ini dibahas pada Bagian 4.4.

Selain angka akhir, dinamika pelatihan pada tiap *epoch* perlu ditinjau untuk memastikan model benar-benar belajar dan bukan sekadar berhenti pada titik yang kebetulan menguntungkan. Tabel 4.3 menyajikan rincian lengkap seluruh 31 *epoch* pada ResNet50 *fold* 0. Baris yang dicetak tebal menandai *epoch* dengan macro-F1 validasi terbaik, dan bobot pada *epoch* itulah yang disimpan sebagai *checkpoint* final model tersebut.

Tabel 4.3 Detail *Epoch* pada ResNet50 *Fold* 0

| *Epoch* | Fase | Laju Pembelajaran | *Loss* Pelatihan | Akurasi Pelatihan (%) | *Loss* Validasi | Akurasi Validasi (%) | Macro-F1 Validasi |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | A | 0,001000 | 0,9189 | 64,24 | 0,6854 | 71,25 | 0,5842 |
| 2 | A | 0,001000 | 0,7973 | 67,19 | 0,6601 | 73,57 | 0,6416 |
| 3 | A | 0,001000 | 0,7712 | 68,62 | 0,6151 | 75,69 | 0,6319 |
| 4 | A | 0,001000 | 0,7459 | 71,73 | 0,7175 | 69,56 | 0,6395 |
| 5 | A | 0,001000 | 0,7163 | 71,52 | 0,6871 | 68,92 | 0,6190 |
| 6 | A | 0,001000 | 0,7051 | 72,20 | 0,6364 | 72,73 | 0,6535 |
| 7 | A | 0,001000 | 0,6847 | 73,00 | 0,6959 | 69,56 | 0,6448 |
| 8 | A | 0,001000 | 0,6751 | 73,79 | 0,5931 | 77,17 | 0,7059 |
| 9 | A | 0,001000 | 0,6670 | 75,63 | 0,6402 | 69,34 | 0,5999 |
| 10 | A | 0,001000 | 0,6554 | 74,58 | 0,6888 | 67,86 | 0,6024 |
| 11 | A | 0,001000 | 0,6489 | 74,53 | 0,6186 | 71,04 | 0,6315 |
| 12 | A | 0,001000 | 0,6473 | 73,68 | 0,5506 | 75,90 | 0,6531 |
| **13** | **B** | **0,000010** | **0,6321** | **76,27** | **0,6104** | **71,04** | **0,6422** |
| 14 | B | 0,000010 | 0,5837 | 76,53 | 0,5708 | 76,11 | 0,7002 |
| 15 | B | 0,000010 | 0,5595 | 78,38 | 0,6087 | 72,73 | 0,6513 |
| 16 | B | 0,000010 | 0,5338 | 79,01 | 0,5674 | 76,32 | 0,7042 |
| 17 | B | 0,000010 | 0,4989 | 80,27 | 0,6035 | 74,21 | 0,6785 |
| 18 | B | 0,000010 | 0,4864 | 80,43 | 0,6249 | 74,84 | 0,6920 |
| 19 | B | 0,000010 | 0,4588 | 80,54 | 0,5838 | 76,11 | 0,7123 |
| 20 | B | 0,000010 | 0,4507 | 81,91 | 0,5711 | 76,32 | 0,7093 |
| 21 | B | 0,000010 | 0,4287 | 82,23 | 0,6030 | 75,69 | 0,7108 |
| 22 | B | 0,000010 | 0,4037 | 82,91 | 0,5083 | 79,07 | 0,7322 |
| 23 | B | 0,000010 | 0,3778 | 84,70 | 0,5149 | 78,86 | 0,7329 |
| 24 | B | 0,000010 | 0,3338 | 85,44 | 0,5349 | 78,44 | 0,7358 |
| **25** | **B** | **0,000010** | **0,3406** | **86,08** | **0,5083** | **80,76** | **0,7565** |
| 26 | B | 0,000010 | 0,3168 | 87,03 | 0,5954 | 77,17 | 0,7264 |
| 27 | B | 0,000010 | 0,2991 | 86,34 | 0,5028 | 79,07 | 0,7391 |
| 28 | B | 0,000010 | 0,2660 | 88,19 | 0,5185 | 78,65 | 0,7303 |
| **29** | **B** | **0,000005** | **0,2667** | **88,66** | **0,4632** | **80,97** | **0,7353** |
| 30 | B | 0,000005 | 0,2421 | 89,72 | 0,5505 | 78,01 | 0,7256 |
| 31 | B | 0,000005 | 0,2495 | 89,56 | 0,5161 | 78,65 | 0,7308 |

Tiga baris pada tabel ini dicetak tebal. Baris *epoch* ke-13 menandai dimulainya fase B beserta turunnya laju pembelajaran dari 0,001000 menjadi 0,000010, baris *epoch* ke-25 menandai macro-F1 validasi terbaik yang bobotnya disimpan sebagai *checkpoint* final, dan baris *epoch* ke-29 menandai penurunan laju pembelajaran oleh *ReduceLROnPlateau*.

Tiga hal terbaca dari tabel ini. Pertama, fase A berhenti pada *epoch* ke-12 sesuai batas maksimumnya, dan macro-F1 validasi tertingginya mencapai 0,7059 pada *epoch* ke-8. Kedua, setelah *fine-tuning* dimulai pada *epoch* ke-13, macro-F1 merangkak naik hingga menembus 0,7565 pada *epoch* ke-25, yaitu kenaikan 0,0506 dari capaian terbaik fase A. Ketiga, mekanisme *ReduceLROnPlateau* menurunkan laju pembelajaran satu kali, yaitu dari 0,000010 menjadi 0,000005 pada *epoch* ke-29, setelah macro-F1 validasi tidak membaik selama tiga *epoch* sejak capaian terbaiknya. Pelatihan berhenti pada *epoch* ke-31 karena *early stopping* terpicu setelah macro-F1 validasi tidak membaik selama enam *epoch* berturut-turut sejak *epoch* ke-25.

Pola pada tabel ini sekaligus memperlihatkan alasan pemilihan *checkpoint* memakai macro-F1 validasi, bukan akurasi validasi. Akurasi validasi tertinggi justru terjadi pada *epoch* ke-29, yaitu 80,97%, sedikit di atas 80,76% pada *epoch* ke-25. Akan tetapi macro-F1 pada *epoch* ke-29 lebih rendah, yang berarti kenaikan akurasi tersebut diperoleh dengan mengorbankan kelas minoritas. Karena penelitian ini menghadapi data yang timpang, kriteria yang dipakai adalah macro-F1 yang memperlakukan ketiga kelas secara setara.


## 4.3 Kontribusi *Fine-Tuning*

Teknik pertama yang diuji adalah *fine-tuning*, yaitu membuka kembali sebagian *backbone* yang semula dibekukan agar bobotnya ikut menyesuaikan diri pada citra CT. Pengukurannya dilakukan dengan membandingkan macro-F1 validasi terbaik yang dicapai pada fase A (hanya lapisan klasifikasi yang dilatih) terhadap macro-F1 validasi terbaik pada fase B (tiga blok terakhir *backbone* ikut dilatih) untuk tiap model. Karena kedua fase dijalankan berurutan pada model dan pembagian data yang sama persis, selisihnya dapat diatribusikan langsung pada *fine-tuning*. Gambar 4.2 menampilkan perubahan tersebut untuk kesepuluh model, dengan arah panah menunjukkan naik atau turunnya macro-F1 setelah fase B dijalankan.

![Gambar 4.2 Kontribusi Fine-Tuning per Model](Gambar/Gambar_4.2_Kontribusi_Finetuning.png)

Gambar 4.2 Perubahan Macro-F1 Validasi dari Fase A ke Fase B pada Kesepuluh Model

Tabel 4.4 menyajikan angkanya secara lengkap.

Tabel 4.4 Macro-F1 Validasi Terbaik Sebelum dan Sesudah *Fine-Tuning*

| Model | Fase A (*feature extraction*) | Fase B (*fine-tuning*) | Selisih |
|---|---:|---:|---:|
| EfficientNet-B0 *fold* 0 | 0,7114 | 0,7060 | −0,0054 |
| EfficientNet-B0 *fold* 1 | 0,6188 | 0,7047 | +0,0858 |
| EfficientNet-B0 *fold* 2 | 0,6822 | 0,7148 | +0,0325 |
| EfficientNet-B0 *fold* 3 | 0,6602 | 0,7194 | +0,0591 |
| EfficientNet-B0 *fold* 4 | 0,6400 | 0,6689 | +0,0290 |
| ResNet50 *fold* 0 | 0,7059 | 0,7565 | +0,0505 |
| ResNet50 *fold* 1 | 0,6563 | 0,7300 | +0,0736 |
| ResNet50 *fold* 2 | 0,7268 | 0,7350 | +0,0082 |
| ResNet50 *fold* 3 | 0,6693 | 0,7301 | +0,0608 |
| ResNet50 *fold* 4 | 0,6313 | 0,6878 | +0,0565 |
| **Rata-rata** | **0,6702** | **0,7153** | **+0,0451** |

Sembilan dari sepuluh model membaik setelah *fine-tuning*, dengan kenaikan rata-rata 0,0451 poin macro-F1. Satu model, yaitu EfficientNet-B0 *fold* 0, justru sedikit menurun (−0,0054). Penurunan tunggal ini tidak dibiarkan merusak hasil akhir karena mekanisme penyimpanan bobot memang menyimpan *checkpoint* dengan macro-F1 validasi tertinggi yang pernah dicapai lintas kedua fase, bukan bobot dari *epoch* terakhir. Dengan kata lain, untuk model tersebut yang tersimpan tetap bobot dari fase A.

Besar manfaat *fine-tuning* tergolong sebanding antar kedua arsitektur. Pada ResNet50, kelima *fold* membaik dengan rata-rata +0,0499, sementara pada EfficientNet-B0 rata-ratanya +0,0402. Selisih 0,0097 di antara keduanya jauh lebih kecil daripada sebaran antar *fold* di dalam masing-masing arsitektur, yang membentang dari +0,0082 hingga +0,0858, sehingga keduanya sebaiknya dibaca sebagai sama-sama terbantu oleh *fine-tuning*, bukan sebagai keunggulan salah satu arsitektur.

Kesimpulan tersebut perlu dikaitkan dengan kapasitas yang benar-benar ikut disesuaikan ketika tiga blok terakhir dibuka, dan besarannya diukur langsung, bukan diperkirakan. Pada ResNet50, tiga blok terakhir mencakup 23,3 juta parameter atau 99,0% dari keseluruhan model. Pada EfficientNet-B0, tiga blok terakhir mencakup 3,2 juta parameter, yang secara proporsi tinggi (78,8%) tetapi secara jumlah hanya sekitar sepertujuh dari ResNet50. Perbedaan ini bersumber pada rancangan EfficientNet-B0 yang memang hemat parameter lewat konvolusi *depthwise separable*, sehingga keseluruhan modelnya hanya berisi 4,0 juta parameter berbanding 23,5 juta pada ResNet50.

Temuan bahwa keduanya terbantu hampir sama besar meski jumlah parameter yang dibuka berbeda tujuh kali lipat menunjukkan bahwa yang menentukan bukanlah banyaknya bobot yang ikut disesuaikan, melainkan terbukanya lapisan tingkat tinggi *backbone* itu sendiri. Lapisan tingkat tinggi ImageNet mengkodekan objek sehari-hari, dan justru lapisan itulah yang paling perlu disesuaikan ketika masukannya berganti menjadi citra CT paru-paru.

Pengaruh tersebut juga terlihat pada kurva pelatihan di Gambar 4.1. Selama fase A, *loss* validasi bergerak naik turun di kisaran 0,5506 hingga 0,7175 dan akurasi validasi berkisar 67,86% hingga 77,17%. Setelah *fine-tuning* dimulai pada *epoch* ke-13, *loss* validasi turun hingga menyentuh 0,4632 dan akurasi validasi naik hingga 80,97%. Pergerakan ini memperlihatkan bahwa sebatas melatih ulang lapisan klasifikasi di atas fitur ImageNet yang beku belum cukup; fitur tingkat tinggi ImageNet perlu disesuaikan dulu pada karakteristik citra CT sebelum dapat dipakai secara optimal.

## 4.4 Kontribusi *Data Augmentation*

Teknik kedua yang diuji adalah augmentasi data. Berbeda dari *fine-tuning* yang selisihnya dapat dibaca langsung dari dua fase pada model yang sama, pengaruh augmentasi hanya dapat diukur dengan melatih ulang seluruh model dari awal memakai *pipeline* yang identik kecuali pada bagian augmentasinya. Karena itu kesepuluh model dilatih ulang sebanyak dua kali lagi, sehingga tersedia tiga kelompok model dengan total tiga puluh model.

Ketiga konfigurasi tersebut adalah sebagai berikut:

1. **Tanpa augmentasi.** Data latih hanya melewati pengubahan ukuran dan normalisasi, sama persis dengan perlakuan pada data validasi dan data uji.
2. **Augmentasi ringan (CT).** Hanya pembalikan horizontal dan rotasi acak hingga 7°, tanpa perubahan kecerahan maupun kontras, tanpa translasi, dan tanpa penskalaan.
3. **Augmentasi penuh.** Pembalikan horizontal, transformasi afin (rotasi 15°, translasi 10%, penskalaan 0,90–1,10), serta perubahan kecerahan dan kontras sebesar 0,2.

Konfigurasi kedua tidak dipilih secara sembarangan, melainkan berangkat dari sifat citra CT itu sendiri. Tingkat keabuan pada citra CT bukan hasil pencahayaan, melainkan berasal dari *Hounsfield Unit* yang merupakan besaran kerapatan jaringan terkalibrasi, sehingga satu tingkat keabuan menandakan satu jenis jaringan tertentu. Mengacak kecerahan dan kontras karena itu tidak sekadar menambah variasi tampilan, melainkan mengubah keterangan kerapatan jaringan yang justru menjadi dasar pembedaan kelas. Pertimbangan serupa berlaku pada transformasi afin: nodul berukuran beberapa milimeter hanya menempati sebagian sangat kecil dari irisan 512×512 piksel, sehingga translasi hingga 10% dan penskalaan berpeluang menggeser atau melarutkan justru objek yang harus dikenali. Konfigurasi ringan dirancang untuk membuang kedua jenis transformasi tersebut dan menyisakan variasi yang memang wajar terjadi, yaitu perbedaan arah hadap pasien dan perbedaan kecil posisi pemindaian.

### 4.4.1 Pengaruh terhadap *Overfitting*

Pengaruh augmentasi paling jelas terlihat bukan pada akurasi, melainkan pada jarak antara performa data latih dan data validasi. Tabel 4.5 menyajikan selisih tersebut pada *epoch* terbaik tiap model, dirata-ratakan atas sepuluh model per konfigurasi.

Tabel 4.5 Pengaruh Kekuatan Augmentasi terhadap *Overfitting* dan Performa Data Uji

| Konfigurasi | Akurasi Latih | Akurasi Validasi | Jurang | Macro-F1 Validasi | Akurasi Data Uji | *Cancer Recall* |
|---|---:|---:|---:|---:|---:|---:|
| Tanpa augmentasi | 89,11% | 77,94% | **+11,17** | 0,7158 | 81,45% | 90,67% |
| Augmentasi ringan (CT) | 82,40% | 77,31% | **+5,09** | 0,7158 | 82,35% | 91,00% |
| Augmentasi penuh | 79,03% | 77,08% | **+1,95** | 0,7013 | 81,67% | 92,33% |

Gambar 4.3 menyajikan kedua sisi temuan ini secara berdampingan, yaitu performa data uji pada panel kiri dan jurang *overfitting* pada panel kanan. Penyajian dua panel dipilih karena satu sumbu saja akan menyesatkan: panel kiri tampak hampir rata, sementara panel kanan menurun tajam.

![Gambar 4.3 Kontribusi Data Augmentation](Gambar/Gambar_4.3_Kontribusi_Augmentasi.png)

Gambar 4.3 Perbandingan Performa Data Uji dan Jurang *Overfitting* pada Tiga Kekuatan Augmentasi

Kolom jurang memperlihatkan pola yang sangat teratur. Tanpa augmentasi, model mencapai 89,11% pada data latihnya sendiri tetapi hanya 77,94% pada data validasi, yaitu selisih 11,17 poin persentase yang menandakan sebagian kemampuannya berupa hafalan atas data latih. Augmentasi ringan memangkas selisih itu menjadi 5,09 poin, dan augmentasi penuh menekannya hingga tinggal 1,95 poin. Urutannya konsisten dengan kekuatan augmentasi, tanpa satu pun penyimpangan.

Perlu dicatat bahwa akurasi validasi ketiga konfigurasi hampir tidak berbeda, yaitu 77,94%, 77,31%, dan 77,08%. Artinya penyempitan jurang tersebut hampir seluruhnya berasal dari turunnya akurasi data latih, bukan dari naiknya akurasi validasi. Augmentasi menahan model agar tidak menghafal, tetapi tidak dengan sendirinya membuatnya lebih pandai pada data baru.

Dengan demikian, augmentasi terbukti menjalankan fungsinya sebagaimana yang diharapkan secara teoretis. Temuan ini berdiri sendiri dan tidak bergantung pada hasil evaluasi akhir.

### 4.4.2 Pengaruh terhadap Performa Data Uji

Pertanyaan berikutnya adalah apakah pengendalian *overfitting* tersebut berbuah performa yang lebih baik pada data uji. Tabel 4.6 membandingkan ketiga konfigurasi pada tiga tingkat agregasi.

Tabel 4.6 Akurasi Data Uji Tiga Konfigurasi Augmentasi pada Tiap Tingkat Agregasi

| Konfigurasi | Model Tunggal (rata-rata 10) | *Soft-Voting* | *Ensemble Stacking* |
|---|---:|---:|---:|
| Tanpa augmentasi | **72,29%** | **74,43%** | 81,45% |
| Augmentasi ringan (CT) | 70,09% | 70,59% | **82,35%** |
| Augmentasi penuh | 70,72% | 71,72% | 81,67% |

Urutan peringkat berubah antar tingkat agregasi tanpa pola yang konsisten. Pada tingkat model tunggal dan *soft-voting*, konfigurasi tanpa augmentasi unggul; pada tingkat *ensemble stacking*, konfigurasi ringan yang di depan sementara tanpa augmentasi justru menjadi yang terendah. Pembalikan urutan semacam ini merupakan tanda bahwa perbedaan yang teramati berada pada taraf fluktuasi acak, bukan pengaruh yang sesungguhnya.

Dugaan tersebut diuji secara formal memakai uji McNemar, yang membandingkan pola benar dan salah dua konfigurasi pada citra uji yang sama. Hasilnya disajikan pada Tabel 4.7.

Tabel 4.7 Uji McNemar antar Konfigurasi Augmentasi pada 442 Citra Uji

| Perbandingan | Hanya A Benar | Hanya B Benar | Nilai p | Kesimpulan |
|---|---:|---:|---:|---|
| Tanpa augmentasi (A) vs ringan CT (B) | 10 | 14 | 0,5413 | tidak berbeda bermakna |
| Tanpa augmentasi (A) vs penuh (B) | 18 | 19 | 1,0000 | tidak berbeda bermakna |
| Augmentasi ringan CT (A) vs penuh (B) | 13 | 10 | 0,6776 | tidak berbeda bermakna |

Tidak satu pun perbandingan menghasilkan nilai p di bawah 0,05. Ketiga konfigurasi karena itu tidak dapat dibedakan secara statistik pada data uji penelitian ini.

Besaran yang memperkuat kesimpulan tersebut adalah perbandingan antara sebaran di dalam satu konfigurasi dengan selisih antar konfigurasi. Simpangan baku akurasi antar sepuluh model di dalam satu konfigurasi berkisar 3,30 hingga 4,05 poin persentase, sementara selisih akurasi terbesar antar konfigurasi hanya 0,90 poin persentase pada tingkat *ensemble stacking*. Dengan kata lain, perbedaan antar *fold* di dalam satu konfigurasi sekitar empat kali lebih besar daripada perbedaan antar konfigurasi yang sedang dibandingkan.

### 4.4.3 Pembahasan

Hasil pada bagian ini perlu dinyatakan apa adanya: **augmentasi data terbukti mengendalikan *overfitting*, tetapi pengendalian tersebut tidak menghasilkan peningkatan performa yang terukur pada data uji penelitian ini.**

Penjelasan yang paling masuk akal atas ketidaksesuaian ini adalah bahwa *overfitting* bukan faktor pembatas utama pada konfigurasi tersebut. Mekanisme penyimpanan bobot yang dipakai penelitian ini hanya menyimpan *checkpoint* dengan macro-F1 validasi tertinggi, bukan bobot dari *epoch* terakhir, sehingga model yang mulai menghafal tidak pernah benar-benar dipakai. Ditambah *early stopping* berbasis macro-F1 validasi dan penggabungan sepuluh model lewat *ensemble*, sebagian besar kerugian akibat hafalan sudah tertangani sebelum augmentasi ikut berperan. Faktor pembatas yang sesungguhnya terletak di tempat lain, yaitu tingkat kesulitan intrinsik subset LIDC-IDRI sebagaimana dibahas pada Bagian 4.8 dan ketidakseimbangan kelas yang menekan kelas Benign.

Karena ketiga konfigurasi tidak dapat dibedakan secara statistik pada data uji, pemilihan konfigurasi final tidak boleh didasarkan pada angka data uji, sebab memilih berdasarkan angka tersebut sama saja dengan memilih berdasarkan derau. Dasar pemilihan yang dipakai adalah performa validasi, yaitu ukuran yang memang tersedia bagi peneliti sebelum data uji dibuka. Pada ukuran itu, konfigurasi ringan dan tanpa augmentasi sama-sama mencapai macro-F1 validasi rata-rata 0,7158, sementara augmentasi penuh tertinggal pada 0,7013. Penentu di antara kedua konfigurasi teratas adalah kestabilannya: simpangan baku akurasi antar sepuluh model pada konfigurasi ringan sebesar 3,30 poin, lebih kecil daripada 4,05 poin pada konfigurasi tanpa augmentasi.

**Konfigurasi augmentasi ringan (CT) karena itu ditetapkan sebagai konfigurasi final** penelitian ini, dan seluruh angka pada Bagian 4.5 sampai Bagian 4.10 dihitung memakai kelompok model tersebut. Sebagai catatan tambahan, konfigurasi ini juga menghasilkan akurasi *ensemble stacking* tertinggi (82,35%) dan macro-F1 tertinggi (0,7228) pada data uji, meski keunggulan tersebut tidak bermakna secara statistik dan karena itu tidak dipakai sebagai dasar pemilihan.

Temuan bahwa augmentasi tidak menaikkan performa secara terukur tetap dilaporkan sebagai hasil penelitian, bukan disembunyikan atau digantikan angka yang lebih menyenangkan. Hasil yang tidak sesuai harapan awal tetap merupakan hasil, dan pada kasus ini justru memberi keterangan yang berguna: pada data CT dengan karakteristik seperti penelitian ini, perbaikan performa lebih menjanjikan bila diarahkan pada penanganan ketidakseimbangan kelas dan kesulitan data, bukan pada penambahan variasi data latih.


## 4.5 Kontribusi *Ensemble Model*

Teknik ketiga adalah penggabungan keluaran beberapa model. Tabel 4.2 memperlihatkan bahwa kesepuluh model tunggal memiliki kekuatan yang berbeda-beda dan tidak selalu keliru pada citra yang sama, sehingga penggabungan berpeluang memperbaiki hasil tanpa perlu melatih model baru sama sekali. Dua pendekatan diuji pada penelitian ini: *soft-voting*, yang merata-ratakan probabilitas dengan bobot tetap, dan *stacking*, yang mempelajari cara menggabungkan dari data. Gambar 4.4 merangkum perjalanan performa dari model tunggal hingga *ensemble stacking*, dan rinciannya dijabarkan pada dua subbagian berikut.

![Gambar 4.4 Kontribusi Ensemble Model](Gambar/Gambar_4.4_Kontribusi_Ensemble.png)

Gambar 4.4 Perbandingan Model Tunggal, *Soft-Voting*, dan *Ensemble Stacking*

### 4.5.1 *Ensemble* *Soft-Voting*

Pendekatan pertama merata-ratakan vektor probabilitas kesepuluh model. Dua varian diuji: bobot setara, di mana setiap model berkontribusi sama besar, dan bobot tertimbang, di mana kontribusi tiap model sebanding dengan macro-F1 validasi yang dicapainya. Hasilnya disajikan pada Tabel 4.8.

Tabel 4.8 Perbandingan Konfigurasi *Soft-Voting*

| Konfigurasi | Anggota | Akurasi | Macro-F1 | *Cancer Recall* | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Rata-rata model tunggal | 1 | 70,09% | 0,663 | 67,63% | 0,896 |
| EfficientNet-B0 saja (5 *fold*) | 5 | 67,19% | 0,649 | 61,67% | 0,897 |
| **ResNet50 saja (5 *fold*)** | 5 | **73,98%** | **0,694** | **74,00%** | **0,913** |
| *Soft-voting* 10 model (bobot setara) | 10 | 70,59% | 0,670 | 68,00% | 0,911 |
| *Soft-voting* 10 model (tertimbang) | 10 | 70,59% | 0,670 | 68,00% | 0,911 |

Penggabungan dengan bobot tetap hanya menaikkan akurasi dari rata-rata 70,09% menjadi 70,59%, yaitu 0,50 poin persentase. Kedua varian pembobotan bahkan menghasilkan prediksi yang sama persis, dan perbedaannya baru muncul pada digit keempat ROC-AUC (0,9111 berbanding 0,9113). Penyebabnya adalah sebaran macro-F1 validasi antar kesepuluh model yang memang rapat, sehingga bobot yang dihasilkan hampir seragam dan hasilnya praktis kembali menjadi rata-rata biasa.

Satu hal yang lebih menonjol terlihat pada baris ketiga: *ensemble* yang hanya berisi kelima model ResNet50 (73,98%) justru mengungguli *ensemble* sepuluh model yang mencampur kedua arsitektur (70,59%) dengan selisih 3,39 poin persentase. Perata-rataan dengan bobot tetap memperlakukan prediksi EfficientNet-B0 yang secara keseluruhan lebih lemah hampir setara dengan prediksi ResNet50, sehingga ikut menarik turun hasil akhir. Dengan kata lain, menambah anggota *ensemble* tidak dengan sendirinya memperbaiki hasil apabila cara penggabungannya tidak membedakan mana anggota yang layak lebih didengarkan pada kasus tertentu. Persoalan inilah yang hendak diselesaikan *stacking*.

### 4.5.2 *Ensemble Stacking*

*Stacking* mengganti bobot tetap dengan sebuah *meta-learner*, yaitu model regresi logistik multinomial yang mempelajari sendiri cara terbaik menggabungkan keluaran kedua arsitektur. Untuk tiap *fold* k, probabilitas keluaran `efficientnet_b0_fold{k}` dan `resnet50_fold{k}` pada data validasi *fold* itu sendiri digabung menjadi satu vektor fitur berdimensi enam (tiga kelas dikali dua arsitektur), lalu dikumpulkan dari seluruh lima *fold* menjadi data latih *meta-learner* sebanyak 2.369 sampel.

Titik kritis metodologisnya terletak pada asal fitur tersebut. Setiap baris data latih *meta-learner* berasal dari citra validasi suatu *fold*, dan diprediksi khusus oleh dua model yang tidak pernah melihat citra itu selama pelatihan. Tanpa pembatasan ini, *meta-learner* akan belajar dari prediksi yang terlalu optimistis dan kenaikan performanya menjadi semu.

*script* kode program konstruksi fitur *out-of-fold*:
```python
meta_X, meta_y = [], []
for k in range(N_FOLDS):
    val_df = trainval[trainval["fold"] == k]
    eff_probs = probs_for_model(predictor, f"efficientnet_b0_fold{k}", val_df["path"].tolist(), transform)
    res_probs = probs_for_model(predictor, f"resnet50_fold{k}", val_df["path"].tolist(), transform)
    feats = np.concatenate([eff_probs, res_probs], axis=1)  # (n, 6)
    meta_X.append(feats)
    meta_y.append(val_df["canonical_label"].map(label_to_idx).values)
    print(f"  fold {k}: {len(val_df)} sampel out-of-fold", flush=True)

meta_X = np.concatenate(meta_X, axis=0)
meta_y = np.concatenate(meta_y, axis=0)
print(f"Total sampel pelatihan meta-learner: {len(meta_X)}", flush=True)

meta_learner = LogisticRegression(max_iter=2000)
meta_learner.fit(meta_X, meta_y)
```

Pada tahap prediksi di data uji, *meta-learner* yang sama diterapkan lima kali (sekali per pasangan model `efficientnet_b0_fold{k}` dan `resnet50_fold{k}`), lalu kelima keluaran probabilitasnya dirata-ratakan menjadi probabilitas akhir. Titik penggabungan antar *fold* dengan demikian bergeser ke belakang *meta-learner*, bukan lagi di depannya seperti pada *soft-voting*. Tabel 4.9 membandingkan kedua pendekatan.

Tabel 4.9 Perbandingan *Soft-Voting* dan *Ensemble Stacking*

| Konfigurasi | Akurasi | Macro-F1 | *Cancer Recall* | *Cancer Precision* |
|---|---:|---:|---:|---:|
| Rata-rata model tunggal | 70,09% | 0,663 | 67,63% | tidak dihitung |
| *Soft-voting* (tertimbang) | 70,59% | 0,670 | 68,00% | tidak dihitung |
| ***Ensemble Stacking* (ambang 0,50)** | **82,35%** | **0,723** | **91,00%** | **88,64%** |

*Stacking* menaikkan akurasi sebesar 11,76 poin persentase dan *cancer recall* sebesar 23,00 poin persentase dibandingkan *soft-voting* tertimbang, tanpa satu pun model dasar dilatih ulang. Dibandingkan rata-rata model tunggal, kenaikannya bahkan mencapai 12,26 poin akurasi dan 23,37 poin *cancer recall*. Seluruh perbaikan ini berasal dari cara kesepuluh keluaran digabungkan, bukan dari model yang lebih baik. Arahnya konsisten dengan pendekatan *ensemble stacking* yang dilaporkan Noman et al. (2025), yang juga menemukan bahwa *meta-learner* yang mempelajari penggabungan dari data cenderung mengungguli agregasi dengan bobot tetap.

Besarnya selisih tersebut dapat dijelaskan lewat temuan pada Bagian 4.2, yaitu bahwa kedua arsitektur memiliki watak yang berbeda. EfficientNet-B0 lebih baik pada kelas Benign sedangkan ResNet50 lebih baik pada kelas Malignant. Perata-rataan dengan bobot tetap memaksa keduanya berbicara sama keras pada setiap kelas, sehingga keunggulan masing-masing saling meredam. *Meta-learner* justru mempelajari pola tersebut dari data dan dapat memberi bobot berbeda untuk tiap kelas, sehingga keunggulan kedua arsitektur dapat dipakai bersamaan alih-alih saling menghapus.

Berbeda dari temuan pada tahap awal penelitian ini, *stacking* kali ini unggul pada ketiga metrik sekaligus, termasuk macro-F1 yang naik 0,0524 poin. Artinya perbaikan tidak lagi terkonsentrasi pada kelas mayoritas semata. Meski begitu, kelas Benign tetap menjadi kelas terlemah sebagaimana terlihat pada *confusion matrix* di Gambar 4.5 dan rincian angka pada Tabel 4.10.

![Gambar 4.5 Confusion Matrix Ensemble Stacking](Gambar/Gambar_4.5_Confusion_Matrix_Stacking.png)

Gambar 4.5 *Confusion Matrix*: *Ensemble Stacking*, Ambang 0,50

Tabel 4.10 Presisi, *Recall*, dan F1-Score per Kelas (*Ensemble Stacking*)

| Kelas | Support | Presisi | *Recall* | F1-Score |
|---|---:|---:|---:|---:|
| Benign | 64 | 0,538 | 0,438 | 0,483 |
| Malignant | 300 | 0,886 | 0,910 | 0,898 |
| Normal | 78 | 0,768 | 0,808 | 0,788 |
| **Rata-rata makro** | **442** | **0,731** | **0,718** | **0,723** |

Kelas Benign hanya menyumbang 13,3% data latih *meta-learner*, dan ketimpangan itu tetap terasa pada hasil akhirnya: dari 64 citra Benign pada data uji, 28 dikenali dengan benar sementara 26 ditebak Malignant dan 10 ditebak Normal. Arah kesalahan yang dominan, yaitu Benign dibaca sebagai Malignant, merupakan arah yang relatif lebih aman dalam konteks skrining dibandingkan kebalikannya, sebab akibatnya berupa pemeriksaan lanjutan yang tidak perlu dan bukan kanker yang terlewat.

Kurva ROC per kelas ditampilkan pada Gambar 4.6, dengan ROC-AUC Malignant 0,927, Normal 0,970, dan Benign 0,835.

![Gambar 4.6 Kurva ROC per Kelas Ensemble Stacking](Gambar/Gambar_4.6_Kurva_ROC_Stacking.png)

Gambar 4.6 Kurva ROC per Kelas: *Ensemble Stacking*

Perbandingan antara ROC-AUC dan *recall* pada kelas Benign menyingkap sesuatu yang tidak terlihat bila hanya salah satunya dilaporkan. ROC-AUC Benign mencapai 0,835, yang berarti model sebenarnya cukup mampu memberi skor lebih tinggi pada citra Benign dibandingkan citra non-Benign. Namun *recall*-nya hanya 0,438. Artinya kemampuan pemeringkatan itu ada, tetapi ambang keputusan tetap (*argmax*) sering dimenangkan kelas lain yang probabilitasnya lebih besar. Persoalan kelas Benign dengan demikian lebih merupakan persoalan kalibrasi dan ketidakseimbangan kelas daripada ketidakmampuan model mengenali polanya sama sekali.

### 4.5.3 Penyesuaian Ambang Keputusan

Sebagai kelanjutan langsung dari temuan di atas, ambang keputusan kelas Malignant pada konfigurasi *stacking* yang sama di-*sweep* dari 0,50 hingga 0,15 tanpa pelatihan ulang. Hasilnya disajikan pada Tabel 4.11 dan Gambar 4.7.

Tabel 4.11 Sweep Ambang Keputusan Kelas Malignant

| Ambang | Akurasi | Macro-F1 | *Cancer Recall* | *Cancer Precision* |
|---|---:|---:|---:|---:|
| **0,50** | 82,35% | **0,723** | 91,00% | **88,64%** |
| 0,40 | **83,48%** | 0,711 | 95,00% | 86,89% |
| 0,35 | 82,81% | 0,653 | 97,67% | 83,95% |
| 0,30 | 81,67% | 0,604 | 98,00% | 82,58% |
| 0,25 | 80,54% | 0,572 | 98,33% | 79,73% |
| 0,20 | 80,54% | 0,564 | 99,33% | 79,05% |
| 0,15 | 79,41% | 0,549 | 99,67% | 77,46% |

![Gambar 4.7 Sweep Ambang Keputusan Ensemble Stacking](Gambar/Gambar_4.7_Sweep_Ambang_Keputusan.png)

Gambar 4.7 Sweep Ambang Keputusan: *Ensemble Stacking*

Akurasi tertinggi justru tercapai pada ambang 0,40 (83,48%), bukan pada ambang *default*. Meski begitu, ambang 0,50 tetap dipilih sebagai konfigurasi final karena memberikan macro-F1 tertinggi (0,723) sekaligus *cancer precision* tertinggi (88,64%). Keunggulan ambang 0,40 pada akurasi hanya 1,13 poin persentase dan diperoleh dengan mengorbankan macro-F1 sebesar 0,012 poin, yang berarti perbaikannya terkonsentrasi pada kelas mayoritas sementara kelas minoritas semakin tertekan. Menurunkan ambang lebih jauh lagi memperlihatkan pertukaran yang semakin curam: pada ambang 0,15, *cancer recall* memang menyentuh 99,67%, tetapi presisinya turun ke 77,46%, yang dalam praktik berarti lebih dari seperlima kasus yang ditandai kanker sebenarnya bukan kanker.

Penelitian ini juga menguji arah sebaliknya, yaitu menaikkan ambang di atas 0,50 agar model lebih ketat sebelum menandai kanker, dan menemukan bahwa mekanisme ambang yang dipakai bersifat satu arah:

```python
def predict_with_threshold(probs, t):
    y_pred = probs.argmax(1)
    override = probs[:, MALIGNANT_IDX] >= t
    return np.where(override, MALIGNANT_IDX, y_pred)
```

Karena fungsi ini hanya *menambahkan* label Malignant lewat `override`, ia tidak pernah *mencabut* label Malignant yang sudah dipilih `argmax` sejak awal, sehingga menaikkan `t` di atas 0,50 tidak mengubah hasil apa pun. Temuan ini dilaporkan apa adanya karena pada awalnya sempat disalahpahami sebagai "ambang tidak berpengaruh", padahal penyebabnya adalah arah mekanismenya, bukan ketidaksensitifan model.

Pengujian lanjutan memakai mekanisme kebalikannya, yang mencabut alih-alih menambah keputusan Malignant, memperjelas pertukaran pada arah tersebut. Pada ambang 0,98, presisi Malignant memang mencapai 100% tetapi *cancer recall* anjlok ke 48,00% dan akurasi keseluruhan turun ke 59,50%. Bahkan pada ambang 0,80 yang jauh lebih longgar, *cancer recall* masih tertahan di 66,00%. Konsekuensi semacam itu tidak sesuai untuk konteks skrining kanker, yang justru menuntut sesedikit mungkin kasus kanker terlewat, sehingga mekanisme ini tidak dipilih.

Berdasarkan seluruh pertimbangan di atas, **konfigurasi final penelitian ini adalah *ensemble stacking* atas kelompok model augmentasi ringan (CT) dengan ambang keputusan 0,50**, dan aplikasi prototipe tetap menyediakan *slider* agar pengguna dapat menurunkan ambang secara manual bila prioritas skrining menuntut sensitivitas lebih tinggi.


## 4.6 Ringkasan Kontribusi Ketiga Teknik

Tabel 4.12 merangkum kontribusi masing-masing teknik yang dinyatakan pada judul penelitian.

Tabel 4.12 Ringkasan Kontribusi Ketiga Teknik Optimasi

| Teknik | Pembanding | Metrik Pembanding | Sebelum | Sesudah | Selisih |
|---|---|---|---:|---:|---:|
| *Fine-tuning* | Fase A vs fase B | Macro-F1 validasi (rata-rata 10 model) | 0,6702 | 0,7153 | +0,0451 |
| *Data augmentation* | Tanpa augmentasi vs konfigurasi final | Akurasi data uji | 81,45% | 82,35% | +0,90 poin (p = 0,54) |
| *Data augmentation* | Tanpa augmentasi vs konfigurasi final | Jurang *overfitting* | +11,17 poin | +5,09 poin | **−6,08 poin** |
| *Ensemble model* | Model tunggal vs *stacking* | Akurasi data uji | 70,09% | 82,35% | **+12,26 poin** |
| *Ensemble model* | Model tunggal vs *stacking* | *Cancer recall* data uji | 67,63% | 91,00% | **+23,37 poin** |

Ketiga teknik yang dinyatakan pada judul memberikan kontribusi yang besarnya sangat berbeda, dan perbedaan itu perlu dinyatakan tanpa dibuat seolah setara.

Kontribusi terbesar datang dari *ensemble model*, khususnya *ensemble stacking*, yang menaikkan akurasi 12,26 poin persentase dan *cancer recall* 23,37 poin persentase dibandingkan rata-rata model tunggal. Besarnya kontribusi ini menegaskan bahwa yang menentukan bukan sekadar menggabungkan banyak model, melainkan cara penggabungannya: *soft-voting* atas kesepuluh model yang sama hanya menghasilkan 70,59%, sementara *meta-learner* atas keluaran yang sama persis mencapai 82,35%.

*Fine-tuning* memberikan kontribusi yang lebih kecil tetapi konsisten, yaitu kenaikan macro-F1 validasi pada sembilan dari sepuluh model dengan rata-rata 0,0451 poin, dan kali ini hampir sama besar pada kedua arsitektur.

*Data augmentation* memberikan hasil yang berbeda sifatnya dari keduanya. Pada metrik performa data uji, pengaruhnya tidak dapat dibedakan dari fluktuasi acak: selisih 0,90 poin persentase dengan nilai p sebesar 0,54 pada uji McNemar, sebagaimana dijabarkan pada Bagian 4.4. Namun pada pengendalian *overfitting*, pengaruhnya justru paling besar di antara ketiga teknik, yaitu memangkas jurang antara akurasi latih dan akurasi validasi dari 11,17 poin menjadi 5,09 poin pada konfigurasi final, bahkan hingga 1,95 poin pada konfigurasi augmentasi penuh. Augmentasi karena itu tidak dapat disebut tidak berkontribusi, melainkan berkontribusi pada aspek yang berbeda dari yang diharapkan semula.

Perlu dicatat pula bahwa ketiga kontribusi tidak dapat dijumlahkan begitu saja, sebab diukur pada tataran yang berbeda: *fine-tuning* pada macro-F1 validasi tiap model, augmentasi pada perbandingan antar kelompok model yang dilatih ulang, dan *ensemble* pada metrik data uji setelah kesepuluh model digabungkan. Model dasar yang digabungkan *ensemble* itu sendiri sudah merupakan model yang melewati *fine-tuning* dan augmentasi, sehingga ketiga kontribusi bersifat berlapis, bukan sejajar.

## 4.7 Analisis Kesalahan Klasifikasi

Dari 442 citra uji pada konfigurasi final, 78 citra (17,6%) masih diklasifikasikan keliru. Tabel 4.13 memecah kesalahan tersebut menurut polanya.

Tabel 4.13 Pola Kesalahan Klasifikasi pada Konfigurasi Final

| Label Asli → Prediksi | Jumlah | Median Keyakinan | Sumber |
|---|---:|---:|---|
| Benign → Malignant | 26 | 0,741 | LIDC-IDRI (26) |
| Malignant → Benign | 18 | 0,494 | LIDC-IDRI (18) |
| Benign → Normal | 10 | 0,579 | LIDC-IDRI (6), Kaggle (4) |
| Malignant → Normal | 9 | 0,506 | LIDC-IDRI (5), Kaggle (4) |
| Normal → Malignant | 9 | 0,548 | LIDC-IDRI (7), Kaggle (2) |
| Normal → Benign | 6 | 0,463 | LIDC-IDRI (3), Kaggle (3) |

Tiga hal menonjol dari sebaran ini. Pertama, 65 dari 78 kesalahan atau 83,3% berasal dari subset LIDC-IDRI, padahal subset tersebut hanya menyumbang 43,2% citra uji. Ketimpangan ini dibahas lebih lanjut pada Bagian 4.8. Kedua, pola kesalahan terbanyak adalah Benign yang ditandai Malignant (26 kasus). Dalam konteks skrining, arah kesalahan ini relatif lebih aman daripada kebalikannya, karena menghasilkan pemeriksaan lanjutan yang tidak perlu alih-alih melewatkan kanker. Ketiga, kesalahan paling berisiko secara klinis, yaitu Malignant yang dianggap Normal, berjumlah 9 kasus atau 3,0% dari seluruh 300 citra Malignant pada data uji.

Angka 3,0% tersebut layak digarisbawahi karena merupakan ukuran yang paling langsung berkaitan dengan tujuan skrining. Sembilan kasus kanker terlewat dari 300 berarti sistem meneruskan 97,0% kasus kanker ke kelas yang setidaknya menuntut tindak lanjut, baik sebagai Malignant maupun sebagai Benign.

Meskipun sebaran kesalahan sudah terpetakan menurut jumlahnya, peninjauan terhadap citra yang gagal diprediksi tetap perlu dilakukan untuk memahami karakteristik visual yang membingungkan model. Dari tiap pola kesalahan pada Tabel 4.13 diambil satu citra, yaitu kesalahan yang diprediksi dengan keyakinan paling tinggi pada pola tersebut. Pemilihan ini disengaja, sebab kesalahan yang diyakini penuh oleh model lebih banyak mengungkap pola yang benar-benar menyesatkan daripada kesalahan yang memang berada di tepi ambang. Seluruh sampel diambil dari prediksi konfigurasi final yang diintegrasikan ke dalam prototipe aplikasi. Setiap analisis disusun dari tiga sumber yang dapat diperiksa ulang, yaitu probabilitas ketiga kelas yang dikeluarkan model, asal label citra, serta pengamatan visual terhadap citranya, sehingga kemungkinan penyebab yang dikemukakan tetap dinyatakan sebagai kemungkinan dan bukan kepastian. Tabel 4.14 berikut menyajikan sampel tersebut beserta analisisnya.

Tabel 4.14 Sampel Kesalahan Prediksi Model

| No | Gambar | Analisis |
|---|---|---|
| 1 | ![4](Gambar/Sampel_Kesalahan/sk_01.png) | Model sangat yakin citra LIDC-IDRI ini tergolong Malignant, dengan probabilitas 97,18%, padahal labelnya Benign. Penelusuran asal label menunjukkan bahwa nodul pada seri ini memiliki skor keganasan rata-rata tepat 3,0, yang berarti para radiolog sendiri tidak sepakat mengenai sifatnya, dan label Benign-nya diperoleh dari pelabelan ulang *nearest-neighbor*, bukan dari penilaian radiolog. Pada irisan setinggi puncak paru ini nodulnya juga tidak tampak jelas, sehingga model kemungkinan bersandar pada karakter keseluruhan irisan LIDC-IDRI, yang pada data latih didominasi kelas Malignant (59,3%). Kasus semacam ini bukan pola umum: dari 26 kesalahan Benign → Malignant, hanya 5 yang labelnya berasal dari pelabelan ulang. Dalam tinjauan metrik evaluasi multikelas, kesalahan ini menyumbang *False Positive* (FP) bagi kelas Malignant dan *False Negative* (FN) bagi kelas Benign, sekaligus tercatat sebagai *True Negative* (TN) bagi kelas Normal. |
| 2 | ![4](Gambar/Sampel_Kesalahan/sk_02.png) | Berbeda dari sampel pertama, model di sini tidak yakin: probabilitas Benign 59,28% berbanding Malignant 35,55%, sedangkan Normal hanya 5,17%. Artinya model mengenali adanya temuan pada citra, tetapi gagal menentukan sifatnya. Label Malignant pada seri ini kuat karena berasal dari skor keganasan rata-rata 4,0 atas empat nodul konsensus. Pembedaan nodul jinak dari nodul ganas bergantung pada ciri seperti bentuk tepi dan ukuran nodul, sementara pada irisan dada utuh 512×512 piksel nodul hanya menempati sebagian kecil piksel. Kesalahan ke arah ini masih tertangani secara klinis, sebab aplikasi tetap menyarankan pemeriksaan lanjutan oleh radiolog untuk hasil Benign. Dalam tinjauan metrik evaluasi multikelas, kesalahan ini menyumbang *False Positive* (FP) bagi kelas Benign dan *False Negative* (FN) bagi kelas Malignant, sekaligus tercatat sebagai *True Negative* (TN) bagi kelas Normal. |
| 3 | ![4](Gambar/Sampel_Kesalahan/sk_03.png) | Model menyatakan citra dari dataset Kaggle ini Normal dengan keyakinan 89,32%, dan secara kasatmata memang tidak tampak lesi pada irisan setinggi puncak paru ini. Label pada dataset tersebut berasal dari nama folder yang disusun pengunggahnya dan berlaku untuk satu kasus pasien, bukan untuk setiap irisan. Karena itu ada kemungkinan irisan ini berasal dari kasus Benign tetapi tidak memuat lesinya, sehingga prediksi Normal justru sesuai dengan isi citranya. Kemungkinan tersebut tidak dapat dipastikan karena dataset ini tidak menyertakan anotasi lokasi lesi. Dalam tinjauan metrik evaluasi multikelas, kesalahan ini menyumbang *False Positive* (FP) bagi kelas Normal dan *False Negative* (FN) bagi kelas Benign, sekaligus tercatat sebagai *True Negative* (TN) bagi kelas Malignant. |
| 4 | ![4](Gambar/Sampel_Kesalahan/sk_04.png) | Sampel ini mewakili kesalahan yang paling berisiko secara klinis karena citra Malignant dinyatakan Normal dengan keyakinan 83,34%. Dua ciri teknis membedakannya dari citra lain. Pertama, resolusi aslinya hanya 240×210 piksel, sehingga ketika diperbesar menjadi 512×512 piksel lebih dari tiga perempat pikselnya merupakan hasil interpolasi, bukan informasi asli dari alat pemindai. Kedua, citra memuat teks ukuran "A: 20.7mm" dan penanda yang tertanam langsung pada citra. Lesi pada citra ini juga tampak berada di dekat hilus, tempat pembuluh darah besar berkumpul, sehingga batasnya berbaur dengan struktur normal di sekitarnya. Dalam tinjauan metrik evaluasi multikelas, kesalahan ini menyumbang *False Positive* (FP) bagi kelas Normal dan *False Negative* (FN) bagi kelas Malignant, sekaligus tercatat sebagai *True Negative* (TN) bagi kelas Benign. |
| 5 | ![4](Gambar/Sampel_Kesalahan/sk_05.png) | Seri LIDC-IDRI ini berlabel Normal karena tidak satu pun radiolog menandai nodul padanya. Model tetap memberi probabilitas Malignant 65,78% dan Benign 26,22%, sehingga lebih dari 90% keyakinannya tertuju pada kelas bernodul. Pada citra tampak percabangan pembuluh darah di sekitar jantung serta area keabuan samar di bagian belakang paru, yang pada irisan tunggal dapat menyerupai nodul kecil. Perlu diingat bahwa label Normal pada LIDC-IDRI berarti tidak ada nodul yang dicatat radiolog, bukan jaminan bahwa paru sepenuhnya bersih dari kelainan. Dalam tinjauan metrik evaluasi multikelas, kesalahan ini menyumbang *False Positive* (FP) bagi kelas Malignant dan *False Negative* (FN) bagi kelas Normal, sekaligus tercatat sebagai *True Negative* (TN) bagi kelas Benign. |
| 6 | ![4](Gambar/Sampel_Kesalahan/sk_06.png) | Keputusan pada citra ini nyaris seperti menebak: probabilitas Benign 48,18% dan Malignant 45,42% hampir setara, dan keyakinan tertingginya pun berada di bawah 50%. Citra ini termasuk rentang keyakinan 0,40–0,55 yang menurut Tabel 4.15 hanya benar pada 47,5% kasus. Struktur yang paling menonjol pada citra adalah pembuluh darah di kedua hilus yang tampak sebagai bintik-bintik terang, dan pada irisan tunggal struktur tersebut sulit dibedakan dari nodul kecil. Seperti sampel kelima, seri ini berlabel Normal karena tidak memiliki nodul yang dicatat radiolog. Dalam tinjauan metrik evaluasi multikelas, kesalahan ini menyumbang *False Positive* (FP) bagi kelas Benign dan *False Negative* (FN) bagi kelas Normal, sekaligus tercatat sebagai *True Negative* (TN) bagi kelas Malignant. |

Keenam sampel memperlihatkan bahwa kesalahan model tidak berasal dari satu penyebab tunggal. Sebagian bersumber dari kualitas citra, seperti resolusi asli yang rendah dan teks yang tertanam pada sampel keempat. Sebagian lain bersumber dari sifat label itu sendiri, seperti label per kasus pada dataset Kaggle dan skor radiolog yang tidak sepakat pada sampel pertama. Sisanya bersumber dari kemiripan visual yang memang sulit dibedakan pada irisan tunggal, terutama antara nodul kecil dan pembuluh darah di sekitar hilus. Temuan ini sejalan dengan analisis per sumber data pada Bagian 4.8, bahwa sebagian besar kesulitan model terkonsentrasi pada subset LIDC-IDRI.

Hubungan antara keyakinan model dan kebenaran prediksinya disajikan pada Tabel 4.15.

Tabel 4.15 Tingkat Kebenaran Prediksi Menurut Rentang Keyakinan Model

| Rentang Keyakinan | Jumlah Citra | Proporsi Prediksi Benar | Porsi terhadap Seluruh Kesalahan |
|---|---:|---:|---:|
| 0,00–0,40 | 8 | 62,5% | 3,8% |
| 0,40–0,55 | 80 | 47,5% | 53,8% |
| 0,55–0,75 | 80 | 78,8% | 21,8% |
| 0,75–0,90 | 70 | 87,1% | 11,5% |
| 0,90–1,00 | 204 | **96,6%** | 9,0% |

Pola ini memiliki nilai praktis yang langsung. Tingkat kebenaran naik secara berurutan seiring naiknya keyakinan, dari 47,5% pada rentang 0,40–0,55 hingga 96,6% pada rentang di atas 0,90. Keteraturan tersebut menunjukkan probabilitas yang dikeluarkan *meta-learner* memang membawa keterangan tentang seberapa layak prediksinya dipercaya, bukan sekadar angka yang kebetulan berada di antara nol dan satu.

Sebaran kesalahannya juga terpusat: 75,6% dari seluruh kesalahan terkumpul pada rentang keyakinan 0,40 hingga 0,75, sementara rentang di atas 0,90 yang mencakup 46,2% citra uji hanya menyumbang 9,0% kesalahan. Dengan kata lain, model umumnya tidak salah dengan penuh keyakinan, melainkan salah ketika memang sedang ragu. Keyakinan prediksi yang ditampilkan aplikasi (Bab III, Bagian 3.5.1) karena itu dapat dipakai sebagai isyarat operasional: prediksi dengan keyakinan di bawah 0,75 sebaiknya ditinjau ulang oleh radiolog, sementara prediksi di atas 0,90 relatif dapat dipercaya.

## 4.8 Analisis Performa per Sumber Data

Angka 82,35% merupakan performa terhadap keseluruhan 442 citra uji yang berasal dari dua sumber dengan karakteristik berbeda. Karena kedua sumber tidak memiliki tingkat kesulitan yang sama, performa model dipecah ulang menurut asal citranya agar angka tunggal tersebut tidak menutupi perbedaan yang ada. Gambar 4.8 memperlihatkan pemisahan tersebut secara visual, dan Tabel 4.16 menyajikan angkanya secara lengkap.

![Gambar 4.8 Performa Konfigurasi Final per Sumber Data](Gambar/Gambar_4.8_Performa_Per_Sumber.png)

Gambar 4.8 Performa Konfigurasi Final Dipecah Menurut Sumber Data

Tabel 4.16 Performa Konfigurasi Final Dipecah Menurut Sumber Data

| Sumber Data Uji | Jumlah Citra | Akurasi | Macro-F1 | *Cancer Recall* | *Benign Recall* |
|---|---:|---:|---:|---:|---:|
| Kaggle | 251 | 94,82% | 0,857 | 97,79% | 66,67% |
| LIDC-IDRI | 191 | 65,97% | 0,564 | 80,67% | 38,46% |
| **Gabungan (angka yang dilaporkan)** | **442** | **82,35%** | **0,723** | **91,00%** | **43,75%** |

Selisihnya sangat besar, yaitu 28,85 poin persentase pada akurasi. Penjelasan yang paling masuk akal terletak pada perbedaan cara kedua dataset dibentuk.

1. **Tingkat kesulitan kasus.** Label pada dataset Kaggle berasal dari nama folder yang disusun pengunggahnya, dan sebagian besar berisi kasus yang secara visual sudah jelas. Sebaliknya, LIDC-IDRI memuat nodul yang dianotasi langsung oleh radiolog beserta skor keganasan 1 sampai 5, termasuk 226 nodul yang skor rata-ratanya tepat di titik tengah karena para radiolog sendiri tidak sepakat. Kasus batas semacam itu memang secara inheren lebih sulit, dan kasus seperti ini justru tidak terwakili pada dataset Kaggle.

2. **Ukuran nodul relatif terhadap citra.** Citra pada kedua sumber sama-sama berupa irisan dada utuh, tetapi kasus LIDC-IDRI dipilih berdasarkan keberadaan nodul beranotasi yang sering kali berukuran hanya beberapa milimeter. Pada irisan berukuran 512×512 piksel, nodul sekecil itu menempati bagian yang sangat kecil dari keseluruhan citra. Kasus Kaggle, yang umumnya sudah menunjukkan massa berukuran besar, jauh lebih mudah dikenali pada skala yang sama.

3. **Keragaman pasien.** Ketujuh dataset Kaggle pada dasarnya merupakan kompilasi ulang dari koleksi yang sama (terutama IQ-OTHNCCD), sehingga meskipun duplikasi persisnya sudah dibuang lewat audit, variasi pasien yang tersisa tetap lebih sempit dibandingkan LIDC-IDRI yang berasal dari 1.010 pasien berbeda.

Angka akurasi pada subset LIDC-IDRI perlu dibaca berdampingan dengan pembanding yang paling sederhana, yaitu strategi yang selalu menebak kelas terbanyak tanpa melihat citra sama sekali. Pada subset tersebut, kelas Malignant mencakup 119 dari 191 citra, sehingga strategi menebak itu saja sudah menghasilkan akurasi 62,30%. Akurasi model 65,97% karena itu hanya 3,67 poin persentase di atasnya, dan bila hanya akurasi yang dilaporkan, model akan tampak nyaris tidak berguna pada subset ini.

Gambaran tersebut berubah ketika dipakai ukuran yang memperlakukan ketiga kelas secara setara. Macro-F1 strategi menebak kelas terbanyak hanya 0,2559, sebab strategi itu benar pada satu kelas dan nol pada dua kelas lainnya, sementara macro-F1 model mencapai 0,5640. Selisih 0,3081 tersebut memperlihatkan bahwa model memang benar-benar membedakan ketiga kelas, bukan sekadar mengikuti kelas terbanyak. Perbedaan kesimpulan antara kedua ukuran ini sekaligus menjadi alasan mengapa penelitian ini memakai macro-F1, bukan akurasi, sebagai kriteria pemilihan *checkpoint* maupun konfigurasi.

Implikasinya untuk pembacaan hasil penelitian ini perlu dinyatakan tanpa ditutup-tutupi. Angka 82,35% sebaiknya dipahami sebagai performa pada populasi campuran, bukan sebagai performa yang akan diperoleh bila sistem dihadapkan pada data klinis nyata. Untuk skenario semacam itu, yang tingkat kesulitannya lebih menyerupai LIDC-IDRI, angka 65,97% merupakan estimasi yang jauh lebih konservatif sekaligus lebih realistis. Penyajian kedua angka secara bersamaan dinilai lebih jujur daripada hanya melaporkan angka gabungan yang terlihat lebih baik.


## 4.9 Implementasi Prototipe Aplikasi

Setelah konfigurasi final ditetapkan, seluruh komponen yang diperlukan untuk menjalankannya diintegrasikan ke dalam prototipe aplikasi *web* berbasis Streamlit. Bagian ini menguraikan model mana yang benar-benar dilayani aplikasi, hasil implementasi model validasi biner yang menjadi lapisan penyaring citra masukan, serta implementasi antarmuka *web* beserta potongan kode programnya.

### 4.9.1 Model yang Diintegrasikan ke Aplikasi

Aplikasi tidak memuat satu berkas model tunggal, melainkan keseluruhan konfigurasi final sebagaimana dievaluasi pada Bagian 4.5.2. Rinciannya adalah sebagai berikut:

1. **Sepuluh model dasar** dari direktori `outputs/models_full_augct`, yaitu `efficientnet_b0_fold0` sampai `efficientnet_b0_fold4` dan `resnet50_fold0` sampai `resnet50_fold4`. Setiap berkas berisi bobot dengan macro-F1 validasi tertinggi yang pernah dicapai model tersebut lintas fase A dan fase B.
2. ***Meta-learner* regresi logistik** (`stacking_meta_learner_full_augct.pkl`) yang menggabungkan keluaran kesepuluh model tersebut, dilatih pada 2.369 sampel *out-of-fold* sebagaimana dijelaskan pada Bagian 4.5.2.
3. **Ambang keputusan 0,50** untuk kelas Malignant, yang dapat digeser pengguna lewat *slider* bila prioritas skrining menuntut sensitivitas lebih tinggi.
4. **Model validasi input** (`ood_detector.pt`), yaitu klasifikasi biner EfficientNet-B0 yang menentukan apakah citra unggahan benar-benar citra CT paru-paru sebelum diteruskan ke model klasifikasi utama.

Aplikasi benar-benar menjalankan alur *stacking*, bukan perata-rataan sederhana. Untuk tiap pasangan *fold*, keluaran EfficientNet-B0 dan ResNet50 disatukan menjadi vektor berdimensi enam, diteruskan ke *meta-learner*, lalu kelima hasilnya dirata-ratakan. Kesesuaian ini diperiksa dengan membandingkan probabilitas keluaran aplikasi terhadap probabilitas yang dipakai menghitung seluruh angka pada bab ini, dan selisih terbesarnya hanya 2,8 × 10⁻¹⁷, yaitu sebatas galat pembulatan bilangan pecahan komputer. Dengan kata lain, angka yang dihasilkan aplikasi identik dengan angka yang dilaporkan, bukan sekadar mendekati.

Resolusi masukan model klasifikasi utama ditetapkan 512×512 piksel agar sama persis dengan resolusi saat model dilatih. Kesesuaian ini wajib dijaga karena memberikan citra beresolusi berbeda kepada model yang di-*fine-tune* pada 512 piksel akan menurunkan akurasi tanpa memunculkan pesan kesalahan apa pun, sehingga kekeliruannya sulit terdeteksi.

### 4.9.2 Implementasi Model Validasi Biner Citra CT Paru-Paru

Tahap implementasi model klasifikasi biner dirancang sebagai mekanisme validasi awal untuk menyaring citra masukan dari pengguna. Pelatihan model ini memanfaatkan total 2.036 citra yang berimbang, terdiri atas 1.018 citra untuk kelas "Citra CT Paru" yang berasal dari seluruh citra CT LIDC-IDRI hasil penyaringan tag `Modality`, dan 1.018 citra untuk kelas "Bukan Citra CT Paru" yang diambil secara acak dari dataset COCO val2017. Pembagian datanya mengikuti pembagian eksperimen utama: citra LIDC-IDRI yang masuk *held-out test set* pada eksperimen utama juga menjadi data uji bagi model ini, sedangkan sisanya dibagi 90:10 menjadi data latih dan data validasi. Dengan cara ini model penyaring tidak pernah diuji memakai citra yang dipakai melatihnya. Rincian distribusi jumlah citra pada masing-masing himpunan data disajikan pada Tabel 4.17.

Tabel 4.17 Pembagian Data pada Validasi Biner Citra CT Paru-Paru

| | *Train* | *Validation* | *Test* |
|---|---:|---:|---:|
| Citra CT Paru | 744 | 83 | 191 |
| Bukan Citra CT Paru | 744 | 83 | 191 |
| **Total** | **1.488** | **166** | **382** |

Model validasi biner memakai arsitektur EfficientNet-B0 *pre-trained* ImageNet dengan seluruh *backbone* dibekukan, sehingga hanya lapisan klasifikasi berkeluaran dua kelas yang dilatih. Citra masukan diubah ukurannya menjadi 224×224 piksel, dengan ukuran *batch* 32 dan *optimizer* Adam berlaju pembelajaran 0,001. Pelatihan dikonfigurasi untuk berjalan paling lama 12 *epoch*, tetapi dihentikan lebih awal secara otomatis begitu akurasi validasi mencapai 99,9%. Pada pelaksanaannya, akurasi validasi sudah mencapai 100% dengan *loss* validasi 0,0514 pada *epoch* pertama, sehingga pelatihan berhenti di situ.

Jumlah *epoch* yang singkat ini sejalan dengan tingkat kesulitan tugasnya yang jauh lebih rendah daripada klasifikasi jenis temuan. Citra CT dada dan citra objek sehari-hari berbeda pada hampir setiap aspek visual, yaitu skala keabuan berbanding warna, struktur anatomi berbanding objek bebas, serta bentuk bingkai melingkar yang khas pada citra CT, sehingga fitur ImageNet yang dibekukan pun sudah cukup untuk memisahkan keduanya.

Evaluasi kinerja akhir model dilakukan pada data uji untuk mengukur keandalannya dalam mengklasifikasikan citra yang belum pernah dilihat sebelumnya. Hasil pemetaan antara prediksi model dan label sebenarnya divisualisasikan melalui *confusion matrix* pada Gambar 4.9.

![Gambar 4.9 Confusion Matrix Validasi Biner](Gambar/Gambar_4.9_Confusion_Matrix_Validasi_Biner.png)

Gambar 4.9 *Confusion Matrix* Validasi Biner Citra CT Paru-Paru

Seluruh 191 citra CT paru dan 191 citra bukan CT paru pada data uji diklasifikasikan dengan benar, tanpa satu pun kesalahan. Keempat metrik evaluasi, yaitu akurasi, presisi, *recall*, dan *F1-Score*, karena itu mencapai nilai 100% pada kedua kelas. Nilai presisi dan *recall* yang sama-sama sempurna menunjukkan model tidak condong ke salah satu kelas, sehingga model ini dinyatakan siap dipasang sebagai gerbang yang menolak unggahan tidak relevan secara otomatis pada antarmuka sistem. Keandalannya terhadap citra yang benar-benar diunggah melalui aplikasi diuji lebih lanjut pada Bagian 4.10.

### 4.9.3 Implementasi *Web*

Implementasi antarmuka pengguna (*User Interface*) dikembangkan memakai bahasa pemrograman Python dengan bantuan kerangka kerja Streamlit. Streamlit dipilih karena mampu mengubah skrip Python menjadi aplikasi *web* interaktif tanpa pemrograman antarmuka terpisah, sehingga model *deep learning* dapat didemonstrasikan secara langsung. Struktur kode program dibagi menjadi empat komponen utama, yaitu pemuatan model, pra-pemrosesan citra, logika inferensi, dan tampilan antarmuka.

a. Inisialisasi dan Pemuatan Model

Komponen pertama adalah fungsi untuk memuat kesepuluh model dasar, *meta-learner*, dan model validasi input. Fungsi pemuatan diberi dekorator `@st.cache_resource` dari Streamlit, sehingga model hanya dimuat satu kali saat aplikasi pertama dijalankan dan disimpan di memori untuk interaksi berikutnya. Tanpa mekanisme ini, sepuluh model beresolusi 512 piksel akan dimuat ulang setiap kali pengguna menggeser *slider* atau mengunggah citra baru, sehingga respons aplikasi menjadi sangat lambat.

*script* kode program pemuatan model:
```python
@st.cache_resource
def load_ood():
    return OODDetector(OOD_MODEL_PATH)


@st.cache_resource
def load_predictor():
    return StackingPredictor(MODEL_DIR, META_PATH, img_size=IMG_SIZE)
```

Di balik kedua fungsi tersebut, kelas `EnsemblePredictor` memuat kesepuluh *checkpoint* satu per satu, membangun ulang arsitekturnya, lalu menyuntikkan bobot terbaik yang disimpan pada tahap pelatihan.

*script* kode program pemuatan *checkpoint* pada EnsemblePredictor:
```python
for arch in self.archs:
    for k in range(N_FOLDS):
        ckpt_path = Path(model_dir) / f"{arch}_fold{k}.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model = build_model(arch).to(self.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        key = f"{arch}_fold{k}"
        self.models[key] = model
        self.val_f1[key] = ckpt.get("best_val_macro_f1", 1.0)
```

Kelas `StackingPredictor` kemudian membungkus `EnsemblePredictor` tersebut dan memuat *meta-learner* regresi logistik dari berkas `.pkl`.

*script* kode program pemuatan *meta-learner*:
```python
class StackingPredictor:
    def __init__(self, model_dir, meta_learner_path, device=None, img_size=512):
        self.base = EnsemblePredictor(model_dir, device=device, img_size=img_size)
        with open(meta_learner_path, "rb") as f:
            self.meta = pickle.load(f)
        self.member_keys = self.base.member_keys
```

b. Pra-pemrosesan Citra Input (*Preprocessing*)

Agar model dapat melakukan prediksi dengan benar, citra yang diunggah pengguna harus melewati transformasi yang identik dengan data validasi dan data uji saat pelatihan. Citra diubah ukurannya menjadi 512×512 piksel, dikonversi menjadi *tensor*, lalu dinormalisasi memakai nilai *mean* dan standar deviasi ImageNet. Augmentasi tidak diterapkan pada tahap ini karena hanya digunakan saat pelatihan.

*script* kode program *preprocessing*:
```python
def build_transforms(image_size: int = 224, train: bool = True, augment_strength: str = "medium") -> v2.Compose:
    """Geometric + intensity augmentation: rotation, horizontal flip, zoom/scale,
    translation, brightness/contrast. Applied ONLY to the training split --
    validation/test always use the deterministic pipeline.
    """
    if not train:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((image_size, image_size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    # ... (cabang augmentasi untuk data latih, lihat Bagian 4.1.1) ...


def _preprocess(self, image):
    x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
    return x
```

c. Logika Inferensi dan Klasifikasi

Inti sistem terletak pada dua tahap prediksi yang dipisahkan secara tegas, yaitu validasi citra CT paru-paru dan klasifikasi jenis temuan. Tahap pertama dijalankan oleh `OODDetector`, yang menghitung probabilitas bahwa citra merupakan CT paru-paru dan menolaknya bila probabilitas tersebut di bawah 0,5. Seluruh perhitungan maju (*forward pass*) dieksekusi di dalam blok `torch.no_grad()` karena model tidak lagi berada dalam fase pelatihan, sehingga memori dan waktu komputasi dapat dihemat.

*script* kode program logika validasi input:
```python
def predict(self, image):
    if not self.available:
        return {"available": False, "is_lung_ct": True, "prob_lung_ct": 1.0}
    x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
    with torch.no_grad():
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
    prob_lung_ct = float(probs[1])
    return {"available": True, "is_lung_ct": prob_lung_ct >= THRESHOLD, "prob_lung_ct": prob_lung_ct}
```

Citra yang lolos validasi diteruskan ke `StackingPredictor`. Fungsi ini menghitung probabilitas *softmax* dari kesepuluh model dasar, menyusun vektor berdimensi enam untuk tiap pasangan *fold*, meneruskannya ke *meta-learner*, lalu merata-ratakan kelima keluaran tersebut. Label akhir ditentukan dengan aturan ambang yang sama persis dengan skrip evaluasi, yaitu kelas Malignant diutamakan begitu probabilitasnya mencapai ambang.

*script* kode program logika klasifikasi *stacking*:
```python
fold_probs = []
for k in range(N_FOLDS):
    eff, res = per_model.get(f"efficientnet_b0_fold{k}"), per_model.get(f"resnet50_fold{k}")
    if eff is None or res is None:
        continue
    feats = np.concatenate([eff, res]).reshape(1, -1)
    fold_probs.append(self.meta.predict_proba(feats)[0])
assert fold_probs, "tidak ada pasangan fold lengkap untuk stacking"

probs = np.mean(fold_probs, axis=0)
# aturan ambang sama dengan stacking_ensemble.py: kelas Malignant
# diutamakan begitu probabilitasnya mencapai ambang
idx = MALIGNANT_IDX if probs[MALIGNANT_IDX] >= threshold else int(probs.argmax())
```

Pada halaman prediksi, kedua tahap tersebut dirangkai berurutan. Bila lapisan validasi menolak citra, pemanggilan `st.stop()` menghentikan eksekusi sehingga model klasifikasi utama tidak pernah dipaksa memberi jawaban atas citra di luar domainnya.

*script* kode program alur validasi dan klasifikasi:
```python
# lapisan validasi input: citra yang bukan CT paru ditolak sebelum
# model klasifikasi utama dijalankan sama sekali
ood = load_ood()
gate = ood.predict(image)
if gate["available"] and not gate["is_lung_ct"]:
    st.error(f"Citra ditolak: terdeteksi BUKAN citra CT paru-paru "
             f"(keyakinan CT paru hanya {gate['prob_lung_ct']*100:.1f}%).")
    st.stop()
if gate["available"]:
    st.success(f"Citra tervalidasi sebagai CT paru-paru "
               f"({gate['prob_lung_ct']*100:.1f}% keyakinan).")

predictor = load_predictor()
result = predictor.predict(image, threshold=ambang)
```

d. Tampilan Antarmuka Pengguna

Antarmuka dirancang sederhana agar mudah dipahami pengguna medis maupun non-medis. Navigasi di sisi kiri memuat enam halaman, yaitu Beranda, Prediksi Citra CT, Perbandingan Model, Kurva *Training*, Audit Dataset, serta Tentang dan Metodologi. Halaman beranda menampilkan metrik konfigurasi final yang dibaca langsung dari berkas hasil evaluasi, sehingga angka yang tampil di aplikasi selalu sama dengan angka yang dilaporkan pada bab ini, yaitu akurasi 82,35%, macro-F1 0,7228, *cancer recall* 91,00%, dan presisi Malignant 88,64%, sebagaimana terlihat pada Gambar 4.10.

![Gambar 4.10 Halaman Beranda Aplikasi](Gambar/Gambar_4.10_Aplikasi_Beranda.png)

Gambar 4.10 Halaman Beranda Prototipe Aplikasi

Interaksi utama berlangsung pada halaman Prediksi Citra CT. Di bagian atas halaman tersedia *slider* ambang keputusan kelas Malignant dengan nilai baku 0,50, diikuti komponen *File Uploader* yang menerima berkas berformat .jpg, .jpeg, .png, dan .bmp. Setelah citra diunggah, sistem menampilkan pratinjaunya, menjalankan lapisan validasi input, lalu menyajikan hasil klasifikasi berupa pesan kelas beserta grafik probabilitas ketiga kelas. Gambar 4.11 memperlihatkan keluaran aplikasi ketika citra CT paru-paru yang sah diunggah: lapisan validasi meloloskan citra dengan keyakinan 98,3%, kemudian model klasifikasi utama menandainya sebagai Malignant.

![Gambar 4.11 Hasil Prediksi Citra CT yang Valid](Gambar/Gambar_4.11_Aplikasi_Prediksi.png)

Gambar 4.11 Hasil Prediksi pada Citra CT Paru-Paru yang Tervalidasi

Di bawah grafik probabilitas tersedia dua panel rincian yang dapat dibuka pengguna, yaitu probabilitas keluaran kesepuluh model dasar dan keluaran *meta-learner* pada kelima pasangan *fold*. Kedua panel ini membuat tahap penggabungan dapat ditelusuri, bukan sekadar menampilkan satu label akhir. Sebaliknya, Gambar 4.12 memperlihatkan respons aplikasi ketika pengguna mengunggah citra objek umum. Lapisan validasi menolak citra tersebut dengan keyakinan CT paru hanya 24,1%, dan proses berhenti sebelum model klasifikasi utama sempat dijalankan.

![Gambar 4.12 Penolakan Citra Bukan CT Paru-Paru](Gambar/Gambar_4.12_Aplikasi_Tolak.png)

Gambar 4.12 Penolakan Aplikasi terhadap Citra Bukan CT Paru-Paru

## 4.10 Evaluasi *Black Box Testing*

Pengujian fungsionalitas sistem dilakukan dengan menerapkan metode *Black Box Testing* untuk memverifikasi bahwa antarmuka aplikasi *web* yang dibangun telah beroperasi sesuai dengan spesifikasi kebutuhan tanpa meninjau kode internal program. Fokus pengujian mencakup mekanisme unggah citra, respons sistem terhadap format berkas, lapisan validasi input, eksekusi *ensemble stacking*, hingga ketepatan tampilan hasil prediksi pada antarmuka pengguna. Sebanyak sembilan skenario uji yang dirancang pada Bab III, Tabel 3.4, dijalankan langsung pada aplikasi yang sudah berjalan melalui otomasi peramban, dan hasil aktualnya dicatat sebagaimana keluaran aplikasi tanpa diketik ulang secara manual. Rincian skenario pengujian beserta perbandingan antara hasil yang diharapkan dan hasil aktual yang diperoleh disajikan pada Tabel 4.18.

Tabel 4.18 Hasil *Black Box Testing*

| No | Fitur | Skenario Pengujian | Hasil yang Diharapkan | Hasil Pengujian (Aktual) | Kesimpulan |
|---|---|---|---|---|---|
| 1 | Widget Pengunggahan (Format Valid) | Mengunggah citra .png dari *held-out test set* | Citra dimuat tanpa pesan kesalahan | Citra berhasil dimuat tanpa pesan kesalahan | 100% Berhasil |
| 2 | Widget Pengunggahan (Format Invalid) | Memeriksa penyaringan jenis berkas pada widget | Sistem menolak berkas bukan citra | Widget hanya menerima .jpg, .jpeg, .png, dan .bmp sehingga berkas .txt tidak dapat dipilih | 100% Berhasil |
| 3 | Area Pratinjau Citra | Memeriksa tampilan citra setelah diunggah | Citra tampil di layar utama | Pratinjau citra tampil sesuai masukan | 100% Berhasil |
| 4 | Validasi Input (Citra CT) | Mengunggah citra CT paru-paru | Diterima dan diteruskan ke klasifikasi | Diterima dengan keyakinan 98,3% | 100% Berhasil |
| 5 | Validasi Input (Bukan CT) | Mengunggah citra objek umum dari dataset COCO | Ditolak sebelum klasifikasi dijalankan | Ditolak dengan keyakinan CT paru 24,1% dan panel klasifikasi tidak muncul | 100% Berhasil |
| 6 | Panel Hasil Prediksi | Menjalankan klasifikasi pada citra valid | Label kelas dan probabilitas tampil | Label Malignant tampil beserta grafik probabilitas ketiga kelas | 100% Berhasil |
| 7 | *Slider* Ambang Keputusan | Menggeser ambang dari 0,50 ke 0,30 | Prediksi dihitung ulang secara konsisten | Ambang berubah ke 0,30, prediksi dihitung ulang, label tetap Malignant | 100% Berhasil |
| 8 | Panel Rincian Anggota *Ensemble* | Membuka rincian probabilitas tiap model dasar | Probabilitas kesepuluh model tampil | Probabilitas kesepuluh model dasar tampil | 100% Berhasil |
| 9 | Panel Rincian *Meta-Learner* | Membuka rincian keluaran *meta-learner* per pasangan *fold* | Probabilitas kelima pasangan *fold* tampil | Keluaran kelima pasangan *fold* tampil | 100% Berhasil |

Berdasarkan hasil pengujian yang tercantum pada Tabel 4.18, dapat disimpulkan bahwa seluruh fitur fungsional utama pada aplikasi *web* klasifikasi kanker paru-paru ini telah beroperasi sesuai dengan skenario yang dirancang. Sistem terbukti mampu menangani berbagai interaksi pengguna, mulai dari memvalidasi format unggahan berkas hingga menolak citra yang tidak sesuai dengan pesan peringatan yang tepat. Fitur inti berupa eksekusi *ensemble stacking* juga berhasil menampilkan label klasifikasi beserta probabilitas ketiga kelas, dan hasilnya dapat dihitung ulang secara langsung ketika pengguna menggeser ambang keputusan. Skenario nomor 9 memperlihatkan pula bahwa tahap penggabungan tingkat kedua benar-benar terjadi di dalam aplikasi, bukan hanya pada skrip evaluasi. Dengan terpenuhinya seluruh kriteria hasil yang diharapkan pada setiap skenario uji, prototipe aplikasi ini dinyatakan lolos uji fungsionalitas.

Untuk membuktikan keandalan sistem dalam menjalankan skenario pengujian nomor 4 dan nomor 5 pada Tabel 4.18, dilakukan pengujian fungsionalitas lanjutan secara khusus terhadap fitur validasi input. Pengujian ini bertujuan melihat secara langsung respons penerimaan maupun penolakan aplikasi ketika dihadapkan pada berbagai jenis citra yang mungkin diunggah pengguna. Pengujian dilakukan dengan mengunggah 20 citra, terdiri atas 12 citra CT paru-paru yang diambil dari *held-out test set*, masing-masing empat citra Malignant, empat citra Benign, dan empat citra Normal, serta 8 citra bukan CT paru-paru berupa objek umum dari dataset COCO, meliputi buah, hewan, peralatan rumah tangga, manusia, kendaraan, dan pemandangan. Kedelapan citra COCO tersebut dipastikan tidak pernah masuk ke manifes pelatihan model validasi biner, sehingga seluruh citra pada pengujian ini belum pernah dilihat model penyaring. Hasil dari pengujian tersebut disajikan selengkapnya pada Tabel 4.19.

Tabel 4.19 Uji Validasi Input

| No. | Gambar | Kategori | *Output* |
|---|---|---|---|
| 1 | ![3.2](Gambar/Validasi_Input/vi_01.png) | Citra CT paru-paru (Malignant). | **Diterima:** Tervalidasi sebagai CT paru-paru (98,2% keyakinan). |
| 2 | ![3.2](Gambar/Validasi_Input/vi_02.png) | Citra CT paru-paru (Malignant). | **Diterima:** Tervalidasi sebagai CT paru-paru (99,5% keyakinan). |
| 3 | ![3.2](Gambar/Validasi_Input/vi_03.png) | Citra CT paru-paru (Malignant). | **Diterima:** Tervalidasi sebagai CT paru-paru (90,1% keyakinan). |
| 4 | ![3.2](Gambar/Validasi_Input/vi_04.png) | Citra CT paru-paru (Malignant). | **Diterima:** Tervalidasi sebagai CT paru-paru (96,7% keyakinan). |
| 5 | ![3.2](Gambar/Validasi_Input/vi_05.png) | Citra CT paru-paru (Benign). | **Diterima:** Tervalidasi sebagai CT paru-paru (98,5% keyakinan). |
| 6 | ![3.2](Gambar/Validasi_Input/vi_06.png) | Citra CT paru-paru (Benign). | **Diterima:** Tervalidasi sebagai CT paru-paru (98,3% keyakinan). |
| 7 | ![3.2](Gambar/Validasi_Input/vi_07.png) | Citra CT paru-paru (Benign). | **Diterima:** Tervalidasi sebagai CT paru-paru (98,0% keyakinan). |
| 8 | ![3.2](Gambar/Validasi_Input/vi_08.png) | Citra CT paru-paru (Benign). | **Diterima:** Tervalidasi sebagai CT paru-paru (99,7% keyakinan). |
| 9 | ![3.2](Gambar/Validasi_Input/vi_09.png) | Citra CT paru-paru (Normal). | **Diterima:** Tervalidasi sebagai CT paru-paru (95,2% keyakinan). |
| 10 | ![3.2](Gambar/Validasi_Input/vi_10.png) | Citra CT paru-paru (Normal). | **Diterima:** Tervalidasi sebagai CT paru-paru (95,8% keyakinan). |
| 11 | ![3.2](Gambar/Validasi_Input/vi_11.png) | Citra CT paru-paru (Normal). | **Diterima:** Tervalidasi sebagai CT paru-paru (95,1% keyakinan). |
| 12 | ![3.2](Gambar/Validasi_Input/vi_12.png) | Citra CT paru-paru (Normal). | **Diterima:** Tervalidasi sebagai CT paru-paru (96,3% keyakinan). |
| 13 | ![3.2](Gambar/Validasi_Input/vi_13.png) | Bukan citra CT paru-paru. | **Ditolak:** Citra terdeteksi BUKAN CT paru-paru (keyakinan CT paru 21,0%). |
| 14 | ![3.2](Gambar/Validasi_Input/vi_14.png) | Bukan citra CT paru-paru. | **Ditolak:** Citra terdeteksi BUKAN CT paru-paru (keyakinan CT paru 1,0%). |
| 15 | ![3.2](Gambar/Validasi_Input/vi_15.png) | Bukan citra CT paru-paru. | **Ditolak:** Citra terdeteksi BUKAN CT paru-paru (keyakinan CT paru 16,2%). |
| 16 | ![3.2](Gambar/Validasi_Input/vi_16.png) | Bukan citra CT paru-paru. | **Ditolak:** Citra terdeteksi BUKAN CT paru-paru (keyakinan CT paru 4,2%). |
| 17 | ![3.2](Gambar/Validasi_Input/vi_17.png) | Bukan citra CT paru-paru. | **Ditolak:** Citra terdeteksi BUKAN CT paru-paru (keyakinan CT paru 3,5%). |
| 18 | ![3.2](Gambar/Validasi_Input/vi_18.png) | Bukan citra CT paru-paru. | **Ditolak:** Citra terdeteksi BUKAN CT paru-paru (keyakinan CT paru 1,2%). |
| 19 | ![3.2](Gambar/Validasi_Input/vi_19.png) | Bukan citra CT paru-paru. | **Ditolak:** Citra terdeteksi BUKAN CT paru-paru (keyakinan CT paru 7,2%). |
| 20 | ![3.2](Gambar/Validasi_Input/vi_20.png) | Bukan citra CT paru-paru. | **Ditolak:** Citra terdeteksi BUKAN CT paru-paru (keyakinan CT paru 21,7%). |

Berdasarkan hasil pengujian yang disajikan pada Tabel 4.19, dapat disimpulkan bahwa mekanisme validasi input berbasis klasifikasi biner telah berfungsi sesuai dengan skenario yang diharapkan. Sistem mengenali dan menerima seluruh 12 citra CT paru-paru untuk diproses lebih lanjut, sekaligus memblokir seluruh 8 citra bukan CT paru-paru dengan menampilkan pesan penolakan yang tepat, sehingga 20 dari 20 citra diputuskan dengan benar. Jarak antara kedua kelompok juga lebar: keyakinan terendah pada citra CT yang diterima adalah 90,1%, sementara keyakinan tertinggi pada citra yang ditolak hanya 21,7%, sehingga gerbang ini tidak bekerja di tepi ambang 0,5. Keberhasilan tersebut menegaskan bahwa lapisan validasi input andal dalam mencegah masuknya citra yang tidak relevan akibat kelalaian pengguna, sehingga integritas proses klasifikasi utama tetap terjaga.

Hasil sempurna ini tetap perlu dibaca secara proporsional. Membedakan citra CT dada dari foto objek sehari-hari jauh lebih mudah daripada membedakan nodul ganas dari nodul jinak, sehingga angka 100% bukan bukti bahwa keseluruhan sistem sempurna, melainkan bahwa lapisan penyaring bekerja andal untuk tugas spesifiknya. Pengujian ini juga belum mencakup kasus batas yang jauh lebih menantang seperti citra CT organ lain, foto rontgen dada, atau citra MRI. Keterbatasan tersebut terasa relevan mengingat temuan pada Bab III, Bagian 3.2.1, bahwa foto rontgen dada memang tersimpan berdampingan dengan citra CT pada koleksi LIDC-IDRI, sehingga pengujian terhadap kasus semacam itu menjadi salah satu saran pengembangan pada Bab V.
