# HALAMAN JUDUL

**OPTIMASI *TRANSFER LEARNING* EFFICIENTNET-B0 UNTUK KLASIFIKASI KANKER PARU-PARU PADA CITRA *COMPUTED TOMOGRAPHY* (CT) MENGGUNAKAN *FINE-TUNING*, *DATA AUGMENTATION*, DAN *ENSEMBLE MODEL***

SKRIPSI

diajukan untuk menempuh ujian sarjana

*(Nama, NPM, Program Studi, Fakultas, Universitas -- isi sesuai data resmi, belum saya isi karena tidak ingin menebak identitas institusional kamu)*

---

## KATA PENGANTAR

*(Bagian ini berisi ucapan terima kasih kepada dosen pembimbing, penguji, dan pihak lain yang membantu penyelesaian skripsi. Saya sengaja tidak mengisi nama-nama di sini karena itu informasi pribadi/institusional yang harus kamu isi sendiri dengan nama pembimbing dan penguji yang sebenarnya.)*

---

## ABSTRAK

Kanker paru-paru merupakan penyebab kematian akibat kanker tertinggi di dunia, dan deteksi dini melalui citra *Computed Tomography* (CT) menjadi faktor penentu keberhasilan pengobatan. Interpretasi manual citra CT menuntut waktu dan ketelitian tinggi, sementara jumlah radiolog terbatas, sehingga sistem bantu klasifikasi otomatis berbasis *deep learning* berpotensi meringankan beban tersebut. Penelitian ini bertujuan mengukur secara terpisah kontribusi tiga teknik optimasi *transfer learning* EfficientNet-B0 untuk klasifikasi citra CT paru-paru ke dalam tiga kelas, yaitu Malignant, Benign, dan Normal.

Penelitian ini menerapkan metode eksperimental dengan menggabungkan tujuh dataset publik Kaggle yang telah diaudit deduplikasinya (1.627 citra berlabel dan 166 citra berlabel perkiraan khusus data latih) dengan dataset LIDC-IDRI. Seri LIDC-IDRI disaring berdasarkan tag `Modality` DICOM karena koleksi tersebut menyimpan 290 seri foto rontgen dada berdampingan dengan citra CT, sehingga tersisa 1.018 citra CT irisan utuh dari 1.010 pasien dengan label yang dikoreksi lewat *nearest-neighbor* dan diagnosis histopatologi TCIA. Dataset gabungan berjumlah 2.811 citra. Model dilatih pada resolusi 512×512 piksel dengan skema *transfer learning* dua fase (*feature extraction* dan *fine-tuning*) dan dievaluasi memakai *Stratified Group K-Fold* lima lipatan. Kontribusi ketiga teknik pada judul diukur terpisah lewat perbandingan terkendali: fase A terhadap fase B untuk *fine-tuning*, tiga kekuatan augmentasi pada tiga puluh model untuk *data augmentation*, serta model tunggal terhadap *soft-voting* dan *ensemble stacking* untuk *ensemble model*.

Konfigurasi final berupa *ensemble stacking* pada ambang 0,50 mencapai akurasi 82,35%, macro-F1 0,7228, *cancer recall* 91,00%, presisi Malignant 88,64%, dan ROC-AUC Malignant 0,9268 pada 442 citra uji yang terpisah sepenuhnya dari pelatihan. Di antara ketiga teknik, *ensemble model* memberikan kontribusi terbesar dengan menaikkan akurasi 12,26 poin persentase dan *cancer recall* 23,37 poin persentase dibandingkan rata-rata model tunggal, sementara *soft-voting* atas kesepuluh model yang sama hanya menaikkan akurasi 0,50 poin. *Fine-tuning* menaikkan macro-F1 validasi pada sembilan dari sepuluh model dengan rata-rata 0,0451 poin. *Data augmentation* terbukti memangkas jurang *overfitting* dari 11,17 poin menjadi 1,95 poin, tetapi pengaruhnya terhadap performa data uji tidak berbeda secara statistik (uji McNemar, p = 0,54 hingga 1,00). Pemecahan hasil menurut sumber data menunjukkan akurasi 94,82% pada subset Kaggle berbanding 65,97% pada subset LIDC-IDRI, sehingga angka gabungan perlu dibaca sebagai performa pada populasi campuran. Sistem ini berpotensi dikembangkan sebagai alat bantu skrining awal, dengan catatan bahwa kelas Benign (*recall* 43,75%) masih menjadi tantangan yang belum terpecahkan.

**Kata Kunci**: *Computed Tomography*, *Data Augmentation*, EfficientNet-B0, *Ensemble Stacking*, *Fine-Tuning*, Kanker Paru-Paru, LIDC-IDRI, *Transfer Learning*

---

## ABSTRACT

Lung cancer is the leading cause of cancer death worldwide, and early detection through Computed Tomography (CT) imaging is a decisive factor in treatment outcomes. Manual interpretation of CT images demands considerable time and precision while the number of radiologists remains limited, so an automated deep-learning classification aid has the potential to ease that burden. This research aims to separately measure the contribution of three optimisation techniques applied to EfficientNet-B0 transfer learning for classifying lung CT images into three classes: Malignant, Benign, and Normal.

This research applies an experimental method by combining seven deduplication-audited public Kaggle datasets (1,627 labelled images and 166 estimated-label images restricted to training data) with the LIDC-IDRI dataset. LIDC-IDRI series were filtered by the DICOM `Modality` tag because the collection stores 290 chest radiograph series alongside CT images, leaving 1,018 whole-slice CT images from 1,010 patients with labels corrected through a nearest-neighbor approach and TCIA histopathology diagnoses. The combined dataset totals 2,811 images. Models were trained at 512×512 pixel resolution using a two-phase transfer learning scheme (feature extraction and fine-tuning) and evaluated with five-fold Stratified Group K-Fold. The contribution of each technique named in the title was measured separately through controlled comparisons: phase A against phase B for fine-tuning, three augmentation strengths across thirty models for data augmentation, and single models against soft-voting and stacking ensemble for ensemble modelling.

The final configuration, a stacking ensemble at a 0.50 threshold, reached an accuracy of 82.35%, macro-F1 of 0.7228, cancer recall of 91.00%, Malignant precision of 88.64%, and Malignant-class ROC-AUC of 0.9268 on 442 held-out test images fully separated from training. Among the three techniques, ensemble modelling contributed the most, raising accuracy by 12.26 percentage points and cancer recall by 23.37 percentage points over the single-model average, whereas soft-voting over the same ten models raised accuracy by only 0.50 points. Fine-tuning improved validation macro-F1 in nine of ten models by an average of 0.0451 points. Data augmentation was shown to cut the overfitting gap from 11.17 to 1.95 percentage points, yet its effect on test performance was not statistically distinguishable (McNemar test, p = 0.54 to 1.00). Breaking the results down by data source revealed 94.82% accuracy on the Kaggle subset against 65.97% on the LIDC-IDRI subset, so the combined figure should be read as performance on a mixed-difficulty population. The system has the potential to be developed as an early screening aid, while acknowledging that the Benign class (43.75% recall) remains an unresolved challenge.

**Keywords**: Computed Tomography, Data Augmentation, EfficientNet-B0, Ensemble Stacking, Fine-Tuning, LIDC-IDRI, Lung Cancer, Transfer Learning