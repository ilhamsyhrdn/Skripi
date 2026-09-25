# HALAMAN JUDUL

**OPTIMASI TRANSFER LEARNING EFFICIENTNET-B0 UNTUK KLASIFIKASI KANKER PARU-PARU PADA CITRA COMPUTED TOMOGRAPHY (CT) MENGGUNAKAN FINE-TUNING, DATA AUGMENTATION, DAN ENSEMBLE MODEL**

SKRIPSI

diajukan untuk menempuh ujian sarjana

*(Nama, NPM, Program Studi, Fakultas, Universitas -- isi sesuai data resmi, belum saya isi karena tidak ingin menebak identitas institusional kamu)*

---

## KATA PENGANTAR

*(Bagian ini berisi ucapan terima kasih kepada dosen pembimbing, penguji, dan pihak lain yang membantu penyelesaian skripsi. Saya sengaja tidak mengisi nama-nama di sini karena itu informasi pribadi/institusional yang harus kamu isi sendiri dengan nama pembimbing dan penguji yang sebenarnya.)*

---

## ABSTRAK

Kanker paru-paru merupakan penyebab kematian akibat kanker tertinggi di dunia, dan deteksi dini melalui citra *Computed Tomography* (CT) menjadi faktor penentu keberhasilan pengobatan. Interpretasi manual citra CT oleh radiolog memakan waktu dan rentan terhadap variasi subjektif, sementara dataset publik yang tersedia untuk melatih model klasifikasi otomatis sering kali bermasalah dari sisi duplikasi data maupun akurasi label. Penelitian ini bertujuan mengimplementasikan dan mengoptimasi model *transfer learning* EfficientNet-B0 untuk klasifikasi citra CT paru-paru ke dalam tiga kelas: Malignant, Benign, dan Normal.

Penelitian ini menerapkan metode eksperimental dengan menggabungkan tujuh dataset publik Kaggle (setelah audit deduplikasi MD5 dan *perceptual hash*, menyisakan 1.627 citra berlabel ditambah 166 citra berlabel perkiraan yang dibatasi hanya untuk data latih) dengan dataset LIDC-IDRI dari *The Cancer Imaging Archive* (1.308 citra irisan dada utuh, label dikoreksi lewat pendekatan *nearest-neighbor* untuk nodul ambigu dan diagnosis histopatologi resmi TCIA), menghasilkan 3.101 citra total. Model dilatih pada resolusi 512×512 piksel dengan skema *transfer learning* dua fase (*feature extraction* dan *fine-tuning*) dan dievaluasi memakai *Stratified Group K-Fold* lima lipatan. Kontribusi ketiga teknik pada judul diukur secara terpisah lewat perbandingan terkendali: fase A dibandingkan fase B untuk *fine-tuning*, tiga kekuatan augmentasi dibandingkan lewat pelatihan ulang tiga puluh model untuk *data augmentation*, serta model tunggal dibandingkan *soft-voting* dan *ensemble stacking* dengan *meta-learner* regresi logistik untuk *ensemble model*.

Konfigurasi final berupa *ensemble stacking* pada ambang 0,50 mencapai akurasi 81,60%, macro-F1 0,682, *cancer recall* 90,35%, presisi Malignant 88,36%, dan ROC-AUC Malignant 0,945 pada 489 citra uji yang terpisah sepenuhnya dari pelatihan. Di antara ketiga teknik, *ensemble model* memberikan kontribusi terbesar dengan menaikkan akurasi 6,61 poin persentase dan *cancer recall* 15,21 poin persentase dibandingkan rata-rata model tunggal, diikuti *fine-tuning* yang menaikkan macro-F1 validasi pada sembilan dari sepuluh model. *Data augmentation* terbukti memangkas jurang *overfitting* dari 10,01 poin menjadi 0,70 poin, tetapi pengaruhnya terhadap performa data uji tidak berbeda secara statistik (uji McNemar, p = 0,80). Pemecahan hasil menurut sumber data menunjukkan akurasi 96,61% pada subset Kaggle berbanding 58,76% pada subset LIDC-IDRI, sehingga angka gabungan perlu dibaca sebagai performa pada populasi campuran. Sistem ini berpotensi dikembangkan sebagai alat bantu skrining awal, dengan catatan bahwa kelas Benign (*recall* 25,0%) masih menjadi tantangan yang belum terpecahkan.

**Kata Kunci**: Computed Tomography, Data Augmentation, EfficientNet-B0, Ensemble Stacking, Fine-Tuning, Kanker Paru-Paru, LIDC-IDRI, Transfer Learning

---

## ABSTRACT

Lung cancer is the leading cause of cancer death worldwide, and early detection through Computed Tomography (CT) imaging is a decisive factor in treatment outcomes. Manual interpretation of CT images by radiologists is time-consuming and prone to subjective variation, while publicly available datasets for training automated classification models are often problematic in terms of both data duplication and label accuracy. This research aims to implement and optimize an EfficientNet-B0 transfer learning model for classifying lung CT images into three classes: Malignant, Benign, and Normal.

This research applies an experimental method by combining seven public Kaggle datasets (after MD5 and perceptual-hash deduplication audit, yielding 1,627 labelled images plus 166 estimated-label images restricted to the training split) with the LIDC-IDRI dataset from The Cancer Imaging Archive (1,308 whole chest-slice images, with ambiguous-nodule labels corrected through a nearest-neighbor approach and further corrected using official TCIA histopathological diagnosis data), resulting in 3,101 images in total. Models were trained at 512x512 resolution using a two-phase transfer learning scheme (feature extraction followed by fine-tuning) and evaluated using 5-fold Stratified Group K-Fold cross-validation. The contribution of each technique named in the title was measured separately through controlled comparisons: phase A against phase B for fine-tuning, three augmentation strengths across thirty retrained models for data augmentation, and single models against soft-voting and stacking ensemble with a logistic-regression meta-learner for ensemble modelling.

The final configuration, a stacking ensemble at a 0.50 threshold, reached an accuracy of 81.60%, macro-F1 of 0.682, cancer recall of 90.35%, Malignant precision of 88.36%, and Malignant-class ROC-AUC of 0.945 on 489 held-out test images fully separated from training. Among the three techniques, ensemble modelling contributed the most, raising accuracy by 6.61 percentage points and cancer recall by 15.21 percentage points over the single-model average, followed by fine-tuning, which improved validation macro-F1 in nine of ten models. Data augmentation was shown to cut the overfitting gap from 10.01 to 0.70 percentage points, yet its effect on test performance was not statistically distinguishable (McNemar test, p = 0.80). Breaking the results down by data source revealed 96.61% accuracy on the Kaggle subset against 58.76% on the LIDC-IDRI subset, so the combined figure should be read as performance on a mixed-difficulty population. The system has the potential to be developed as an early screening aid, while acknowledging that the Benign class (25.0% recall) remains an unresolved challenge.

**Keywords**: Computed Tomography, Data Augmentation, EfficientNet-B0, Ensemble Stacking, Fine-Tuning, LIDC-IDRI, Lung Cancer, Transfer Learning
