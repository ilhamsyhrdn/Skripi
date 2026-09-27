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

Kanker paru-paru merupakan penyebab kematian akibat kanker tertinggi di dunia, dan deteksi dini melalui citra *Computed Tomography* (CT) sangat menentukan keberhasilan pengobatan. Namun, interpretasi manual citra CT memakan waktu dan bergantung pada pengalaman radiolog, karena satu pemindaian dapat menghasilkan ratusan irisan yang harus ditelaah satu per satu. Penelitian ini bertujuan mengoptimasi *transfer learning* EfficientNet-B0 menggunakan *fine-tuning*, *data augmentation*, dan *ensemble model* untuk mengklasifikasikan citra CT paru-paru ke dalam tiga kelas, yaitu Malignant, Benign, dan Normal.

Penelitian ini menerapkan metode eksperimental dengan menggabungkan tujuh dataset publik Kaggle dan dataset LIDC-IDRI yang telah disaring agar hanya memuat citra CT, sehingga menghasilkan total 2.811 citra. Tahapan komputasi meliputi *preprocessing* berupa *windowing* HU pada berkas DICOM dan penyeragaman ukuran citra menjadi 512×512 piksel, serta augmentasi data latih berupa pembalikan horizontal dan rotasi ringan. Model dilatih menggunakan pendekatan *transfer learning* dua fase, yaitu *feature extraction* dan *fine-tuning*, dengan *optimizer* Adam, kemudian prediksi EfficientNet-B0 dan ResNet50 digabungkan melalui *ensemble stacking*. Evaluasi performa dilakukan menggunakan *Stratified Group K-Fold* lima lipatan dan 442 citra uji yang terpisah sepenuhnya dari data latih.

Hasil pengujian menunjukkan bahwa konfigurasi final mencapai akurasi 82,35%, macro-F1 0,7228, *cancer recall* 91,00%, dan presisi Malignant 88,64%. *Ensemble stacking* memberikan kontribusi terbesar dengan menaikkan akurasi 12,26 poin persentase dibandingkan rata-rata model tunggal, *fine-tuning* meningkatkan macro-F1 validasi pada sembilan dari sepuluh model, dan *data augmentation* menyempitkan jurang *overfitting* dari 11,17 poin hingga 1,95 poin persentase. Kesimpulannya, model yang dihasilkan mampu mengenali sebagian besar kasus kanker paru-paru dan berpotensi dikembangkan sebagai sistem *Computer-Aided Diagnosis* (CAD) untuk membantu radiolog, meskipun klasifikasi kelas Benign masih perlu ditingkatkan.

**Kata Kunci**: *Computed Tomography*, *Data Augmentation*, EfficientNet-B0, *Ensemble Model*, *Fine-Tuning*, Kanker Paru-Paru, *Transfer Learning*

---

## ABSTRACT

Lung cancer is the leading cause of cancer death worldwide, and early detection through Computed Tomography (CT) imaging largely determines treatment success. However, manual interpretation of CT images is time-consuming and depends on the radiologist's experience, since a single scan can produce hundreds of slices that must be reviewed one by one. This research aims to optimize EfficientNet-B0 transfer learning using fine-tuning, data augmentation, and ensemble modeling to classify lung CT images into three classes: Malignant, Benign, and Normal.

This research applies an experimental method by combining seven public Kaggle datasets with the LIDC-IDRI dataset, which was filtered to contain CT images only, yielding a total of 2,811 images. The computational stages include preprocessing through HU windowing of DICOM files and resizing images to 512×512 pixels, as well as augmenting the training data with horizontal flipping and slight rotation. The models were trained using a two-phase transfer learning approach, namely feature extraction and fine-tuning, with the Adam optimizer, and the predictions of EfficientNet-B0 and ResNet50 were then combined through a stacking ensemble. Performance was evaluated using five-fold Stratified Group K-Fold and 442 test images fully separated from the training data.

Test results show that the final configuration achieved an accuracy of 82.35%, a macro-F1 of 0.7228, a cancer recall of 91.00%, and a Malignant precision of 88.64%. The stacking ensemble made the largest contribution, raising accuracy by 12.26 percentage points over the single-model average, fine-tuning improved validation macro-F1 in nine of ten models, and data augmentation narrowed the overfitting gap from 11.17 to 1.95 percentage points. In conclusion, the resulting model is able to recognize most lung cancer cases and has the potential to be developed into a Computer-Aided Diagnosis (CAD) system to assist radiologists, although classification of the Benign class still needs improvement.

**Keywords**: Computed Tomography, Data Augmentation, EfficientNet-B0, Ensemble Model, Fine-Tuning, Lung Cancer, Transfer Learning