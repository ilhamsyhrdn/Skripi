# BAB I

# PENDAHULUAN

Pada bab ini dijelaskan latar belakang topik penulisan skripsi, identifikasi dan batasan masalah, maksud dan tujuan penelitian, manfaat penelitian, metodologi yang digunakan, serta sistematika penulisan.

## 1.1 Latar Belakang

Kanker paru-paru adalah penyebab kematian akibat kanker tertinggi di dunia, melampaui gabungan kanker payudara, prostat, dan usus besar. Tingkat kelangsungan hidup lima tahun pasien kanker paru-paru yang terdiagnosis pada stadium lanjut jauh lebih rendah dibandingkan pasien yang terdeteksi pada stadium dini, sehingga deteksi dini menjadi faktor yang menentukan peluang kesembuhan. *Computed Tomography* (CT) *scan* merupakan modalitas pencitraan yang paling umum digunakan untuk skrining dan diagnosis kelainan pada paru-paru karena mampu menampilkan struktur nodul sekecil beberapa milimeter yang tidak selalu terlihat pada foto rontgen dada biasa.

Meskipun demikian, interpretasi citra CT oleh radiolog tetap merupakan pekerjaan yang menuntut, memakan waktu, dan bergantung pada pengalaman individu. Satu pemindaian CT dada dapat menghasilkan ratusan irisan gambar yang harus ditelaah satu per satu untuk mencari nodul yang dicurigai ganas, dan keputusan akhir mengenai jinak atau ganasnya suatu nodul sering kali baru bisa dipastikan lewat biopsi. Keterbatasan tenaga radiolog di banyak fasilitas kesehatan, ditambah risiko kelelahan visual pada pembacaan citra dalam jumlah besar, membuka ruang bagi sistem bantu diagnosis otomatis berbasis *Deep Learning* sebagai alat bantu skrining awal, bukan pengganti keputusan klinis.

*Convolutional Neural Network* (CNN) telah berulang kali terbukti efektif untuk tugas klasifikasi citra medis, termasuk citra CT paru-paru. Salah satu arsitektur CNN yang dipilih dalam penelitian ini adalah EfficientNet-B0, anggota terkecil dari keluarga EfficientNet yang diperkenalkan Tan & Le (2019). Keunggulan EfficientNet terletak pada strategi *compound scaling*, yaitu menaikkan kedalaman, lebar, dan resolusi jaringan secara serentak dengan rasio tetap, sehingga model mencapai akurasi tinggi tanpa membengkakkan jumlah parameter sebagaimana arsitektur CNN konvensional. Untuk mengatasi keterbatasan jumlah data medis berlabel, penelitian ini menerapkan *transfer learning*: bobot EfficientNet-B0 yang telah dilatih pada ImageNet dipakai sebagai titik awal, kemudian disesuaikan lewat *fine-tuning* pada citra CT paru-paru.

Tantangan yang justru lebih besar daripada pemilihan arsitektur adalah kualitas data itu sendiri. Sebagian besar dataset citra CT paru-paru yang beredar bebas di Kaggle merupakan kompilasi ulang dari sumber yang sama, disusun oleh pengunggah yang berbeda-beda tanpa metadata pasien yang jelas, sehingga rawan duplikasi lintas dataset maupun duplikasi di dalam dataset itu sendiri. Duplikasi semacam ini, bila tidak dideteksi sebelum data displit ke *train* dan *test*, menyebabkan *data leakage*: model seolah-olah mencapai akurasi tinggi karena "menghafal" gambar yang identik pernah dilihat saat pelatihan, bukan karena benar-benar belajar mengenali pola nodul. Selain itu, sebagian dataset ini memberi label berdasarkan nama folder yang ditulis penyusun dataset, bukan berdasarkan diagnosis histopatologi yang dikonfirmasi laboratorium, sehingga akurasi label itu sendiri patut dipertanyakan.

Untuk mengatasi celah tersebut, penelitian ini tidak berhenti pada satu sumber data. Selain mengumpulkan dan mengaudit tujuh dataset publik dari Kaggle (dideduplikasi lewat pencocokan MD5 dan *perceptual hash* sebelum digabung), penelitian ini juga menambahkan LIDC-IDRI (*Lung Image Database Consortium and Image Database Resource Initiative*), koleksi data CT resmi dari *National Cancer Institute* yang didistribusikan lewat *The Cancer Imaging Archive* (TCIA) dan disertai anotasi nodul oleh hingga empat radiolog independen per kasus (Armato et al., 2011). Setiap nodul pada LIDC-IDRI diberi skor keganasan 1–5 oleh masing-masing radiolog, dan pada sebagian pasien tersedia pula data diagnosis histopatologi resmi dari TCIA yang menjadi rujukan koreksi label bila skor radiolog bersifat ambigu. Nodul dengan skor keganasan yang persis di titik tengah (skor rata-rata 3, yang berarti radiolog sendiri tidak sepakat apakah nodul itu jinak atau ganas) dilabeli ulang lewat pendekatan *nearest-neighbor* pada ruang fitur visual, mengikuti metode yang diusulkan Zhang et al. (2022) untuk kasus serupa pada LIDC-IDRI, bukan ditebak atau dibuang begitu saja.

Gabungan strategi *fine-tuning*, augmentasi data, dan *ensemble model* (tiga istilah yang tercermin langsung pada judul penelitian ini) diuji secara bertahap: mulai dari model tunggal pada tiap sumber data, *ensemble* sederhana lewat *soft-voting* antara EfficientNet-B0 dan ResNet50 pada lima *fold*, hingga *ensemble stacking* di mana sebuah *meta-learner* mempelajari cara terbaik menggabungkan keluaran kedua arsitektur tersebut alih-alih sekadar merata-ratakannya. Setiap tahap dibandingkan secara jujur terhadap tahap sebelumnya berdasarkan akurasi, *cancer recall* (proporsi kasus kanker yang benar-benar tertangkap oleh model, metrik yang secara klinis lebih krusial daripada akurasi semata), dan ROC-AUC, sehingga klaim performa akhir penelitian ini bisa ditelusuri ke eksperimen pembandingnya, bukan angka yang berdiri sendiri.

## 1.2 Identifikasi Masalah

Berdasarkan uraian latar belakang, masalah yang dicari solusinya dalam penelitian ini adalah sebagai berikut:

1. Bagaimana menerapkan *transfer learning* EfficientNet-B0 untuk mengklasifikasikan citra CT paru-paru ke dalam tiga kategori: Malignant (ganas), Benign (jinak), dan Normal?
2. Bagaimana mendeteksi dan menangani duplikasi citra, baik di dalam satu dataset maupun lintas dataset publik, agar evaluasi model tidak bias akibat *data leakage*?
3. Bagaimana menangani nodul pada dataset LIDC-IDRI yang skor keganasannya ambigu (tidak disepakati oleh radiolog) tanpa membuang data tersebut begitu saja?
4. Seberapa besar kontribusi masing-masing teknik optimasi, yaitu *fine-tuning*, *data augmentation*, dan *ensemble model*, terhadap peningkatan akurasi dan *cancer recall* dibandingkan model dasar tanpa teknik tersebut?

## 1.3 Batasan Masalah

Penelitian ini dibatasi pada hal-hal berikut:

1. Penelitian berfokus pada *transfer learning* dan *fine-tuning* atas model *pre-trained*, bukan melatih arsitektur CNN dari nol.
2. Arsitektur utama yang dievaluasi adalah EfficientNet-B0, dengan ResNet50 digunakan sebagai arsitektur kedua khusus untuk keperluan *ensemble*, bukan sebagai perbandingan independen yang setara.
3. Klasifikasi dibatasi pada tiga kelas pada level citra (irisan CT 2D): Malignant, Benign, dan Normal. Penelitian ini tidak melakukan segmentasi nodul maupun deteksi lokasi nodul pada citra.
4. Data yang digunakan seluruhnya berasal dari dataset publik yang telah tersedia secara daring: tujuh dataset dari Kaggle dan satu dataset dari TCIA (LIDC-IDRI), tidak ada pengambilan data primer dari rumah sakit atau pasien secara langsung.
5. Pelatihan dibatasi maksimal 12 *epoch* untuk fase *feature extraction* dan 25 *epoch* untuk fase *fine-tuning* per model, dengan *early stopping* berdasarkan macro-F1 validasi, mengingat keterbatasan sumber daya komputasi yang tersedia.
6. Evaluasi kinerja dibatasi pada metrik kuantitatif: akurasi, presisi, *recall* per kelas (dengan penekanan pada *cancer recall*), F1-*score*, dan ROC-AUC.
7. Penelitian ini menyertakan prototipe aplikasi *web* sederhana berbasis Streamlit untuk mendemonstrasikan hasil klasifikasi dan *ensemble* secara interaktif, dilengkapi lapisan validasi input yang menolak citra di luar domain, bukan sebagai sistem produksi yang siap dipakai di lingkungan klinis nyata.

## 1.4 Maksud dan Tujuan Penelitian

Maksud dari penelitian ini adalah mengimplementasikan dan mengoptimasi model *transfer learning* EfficientNet-B0 untuk klasifikasi kanker paru-paru pada citra CT, dengan data yang telah diaudit dan dikoreksi labelnya secara metodologis.

Tujuan yang ingin dicapai adalah:

1. Mengimplementasikan model *transfer learning* EfficientNet-B0 dengan skema dua fase (*feature extraction* lalu *fine-tuning*) untuk klasifikasi citra CT paru-paru ke dalam kelas Malignant, Benign, dan Normal.
2. Membangun jalur audit data yang mendeteksi duplikasi lintas dan di dalam dataset Kaggle, serta memproses dataset LIDC-IDRI dari berkas DICOM dan anotasi XML radiolog menjadi label kelas yang dapat dipakai untuk pelatihan.
3. Menerapkan koreksi label pada nodul LIDC-IDRI yang ambigu (skor keganasan rata-rata radiolog tepat di titik tengah) melalui pendekatan *nearest-neighbor* pada ruang fitur visual, serta melalui data diagnosis histopatologi resmi TCIA bila tersedia.
4. Mengukur kontribusi *fine-tuning*, *data augmentation*, dan *ensemble model* secara terpisah lewat perbandingan terkendali, lalu menentukan konfigurasi akhir dengan keseimbangan akurasi dan *cancer recall* terbaik.
5. Membangun prototipe aplikasi *web* yang mendemonstrasikan seluruh alur di atas, dilengkapi lapisan validasi input berbasis klasifikasi biner yang menolak citra di luar domain sebelum diproses model utama.

## 1.5 Manfaat Penelitian

Manfaat yang diharapkan dari penelitian ini adalah:

1. Memberikan kontribusi metodologis mengenai cara mengaudit dan menggabungkan dataset citra medis dari sumber yang heterogen (dataset Kaggle berbasis nama folder dan dataset akademik berbasis anotasi radiolog) tanpa mengorbankan validitas evaluasi akibat *data leakage*.
2. Menjadi referensi bagi peneliti lain yang ingin menangani masalah label ambigu pada dataset LIDC-IDRI tanpa membuang data yang ambigu tersebut.
3. Menghasilkan model klasifikasi yang berpotensi membantu skrining awal nodul paru-paru pada citra CT, dengan penekanan pada *cancer recall* yang tinggi sebagai prioritas klinis.
4. Menyediakan prototipe aplikasi yang dapat dipakai sebagai alat peraga untuk menjelaskan cara kerja model klasifikasi citra medis kepada audiens non-teknis.

## 1.6 Metodologi Penelitian

Penelitian ini menggunakan metode eksperimental dengan tahapan sebagai berikut:

1. **Studi Literatur.** Mengumpulkan referensi mengenai kanker paru-paru, interpretasi citra CT, arsitektur EfficientNet, ResNet, *transfer learning*, serta metode penanganan label ambigu pada LIDC-IDRI (Zhang et al., 2022).
2. **Pengumpulan dan Audit Data.** Mengunduh tujuh dataset Kaggle dan dataset LIDC-IDRI beserta anotasi XML radiolognya dari TCIA. Setiap dataset Kaggle diaudit lewat pencocokan MD5 dan *perceptual hash* untuk mendeteksi duplikasi di dalam maupun lintas dataset, serta pemetaan label dari struktur folder ke tiga kelas target.
3. **Pemrosesan Data LIDC-IDRI.** Membaca tag `Modality` pada kepala tiap berkas DICOM untuk memisahkan seri *Computed Tomography* dari foto rontgen dada yang tersimpan berdampingan dalam struktur folder yang sama, lalu mem-parsing anotasi XML tiap radiolog, mengelompokkan penanda nodul yang saling berdekatan menjadi "nodul konsensus", menghitung skor keganasan rata-rata per nodul, lalu menentukan label seri (Malignant/Benign/Excluded jika skor tepat di titik tengah, atau Normal jika tidak ada nodul yang tercatat). Citra diekstraksi dari DICOM lewat *windowing* HU (*Hounsfield Unit*) pada bentuk irisan dada utuh 512×512 piksel, seragam dengan bentuk citra pada dataset Kaggle.
4. **Koreksi Label Ambigu.** Nodul dengan label "Excluded" dilabeli ulang lewat pencarian *nearest-neighbor* pada ruang fitur EfficientNet-B0 *pre-trained* terhadap nodul yang labelnya sudah pasti (Malignant/Benign), kemudian dikoreksi lebih lanjut memakai data diagnosis histopatologi resmi TCIA pada pasien yang datanya tersedia.
5. **Penggabungan Dataset.** Menggabungkan pool Kaggle dan pool LIDC-IDRI yang sudah diberi label final, disertai pengecekan ulang duplikasi lintas kedua sumber, sebelum data displit memakai *Stratified Group K-Fold* (grup berdasarkan pasien/kasus asal, bukan citra individual) untuk mencegah kebocoran data.
6. **Pelatihan Model.** Melatih EfficientNet-B0 dan ResNet50 pada lima *fold*, masing-masing melalui dua fase: *feature extraction* (*backbone* dibekukan) lalu *fine-tuning* (beberapa blok terakhir *backbone* dibuka), dengan augmentasi data (*flip*, rotasi, translasi, skala, perubahan warna) hanya pada data latih.
7. **Pengukuran Kontribusi Tiap Teknik.** Mengukur sumbangan *fine-tuning*, *data augmentation*, dan *ensemble model* secara terpisah lewat perbandingan terkendali: fase A melawan fase B untuk *fine-tuning*, pelatihan ulang tanpa augmentasi untuk *data augmentation*, serta model tunggal melawan *soft-voting* dan *stacking* untuk *ensemble*, seluruhnya pada data uji yang belum pernah dilihat selama pelatihan maupun validasi.
8. **Pembangunan Prototipe Aplikasi.** Mengimplementasikan model terbaik ke dalam aplikasi Streamlit untuk mendemonstrasikan hasil klasifikasi.
9. **Pelaporan.** Mendokumentasikan seluruh proses dan hasil ke dalam laporan skripsi secara sistematis dan jujur, termasuk eksperimen yang hasilnya kurang memuaskan.

## 1.7 Sistematika Penulisan

Sistematika penulisan skripsi ini disusun sebagai berikut:

**BAB I PENDAHULUAN**
Berisi latar belakang, identifikasi dan batasan masalah, maksud dan tujuan penelitian, manfaat penelitian, metodologi, dan sistematika penulisan.

**BAB II TINJAUAN PUSTAKA**
Berisi landasan teori mengenai kanker paru-paru, citra CT, *Deep Learning* dan CNN, *transfer learning*, arsitektur EfficientNet dan ResNet, teknik optimisasi dan regularisasi, metode penanganan dataset LIDC-IDRI, perangkat lunak yang digunakan, penelitian terkait, serta metode evaluasi sistem.

**BAB III ANALISIS DAN PERANCANGAN**
Berisi alur penelitian, analisis kebutuhan data (termasuk rincian audit dan pemrosesan tiap sumber dataset), kebutuhan perangkat keras dan lunak, perancangan skenario eksperimen, perancangan model, perancangan prototipe aplikasi, serta rancangan pengujian fungsionalitas aplikasi dengan *Black Box Testing*.

**BAB IV HASIL DAN PEMBAHASAN**
Berisi hasil pengukuran kontribusi ketiga teknik yang dinyatakan pada judul, yaitu *fine-tuning*, *data augmentation*, dan *ensemble model*, masing-masing lewat perbandingan terkendali pada data uji yang sama. Dilengkapi penyesuaian ambang keputusan, analisis kesalahan klasifikasi, pemecahan performa menurut sumber data, implementasi model validasi biner dan prototipe aplikasi *web*, serta hasil *Black Box Testing* dan uji validasi input.

**BAB V KESIMPULAN DAN SARAN**
Berisi kesimpulan yang menjawab identifikasi masalah pada Bab I, serta saran untuk pengembangan penelitian selanjutnya.
