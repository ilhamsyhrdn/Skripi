# BAB I

# PENDAHULUAN

Pada bab ini dijelaskan latar belakang topik penulisan skripsi, pokok permasalahan berupa identifikasi dan batasan masalah, maksud dan tujuan penelitian, manfaat yang diharapkan dari penelitian, metodologi yang digunakan, serta sistematika penulisan skripsi.

## 1.1 Latar Belakang

Kanker paru-paru merupakan kanker yang paling banyak didiagnosis sekaligus penyebab kematian akibat kanker tertinggi di dunia. Pada tahun 2022 tercatat hampir 2,5 juta kasus baru kanker paru-paru atau 12,4% dari seluruh kasus kanker, serta sekitar 1,8 juta kematian atau 18,7% dari seluruh kematian akibat kanker (Bray et al., 2024). Beban sebesar itu menjadikan deteksi dini sebagai faktor penentu, sebab skrining memakai *Computed Tomography* (CT) dosis rendah pada kelompok berisiko tinggi terbukti menurunkan kematian akibat kanker paru-paru sebesar 20% (Duma et al., 2019). CT *scan* menjadi modalitas pencitraan yang paling umum dipakai untuk skrining dan diagnosis kelainan paru-paru karena mampu menampilkan struktur nodul berukuran beberapa milimeter yang tidak selalu terlihat pada foto rontgen dada biasa.

Meskipun demikian, interpretasi citra CT oleh radiolog tetap merupakan pekerjaan yang menuntut, memakan waktu, dan bergantung pada pengalaman individu. Satu pemindaian CT dada dapat menghasilkan ratusan irisan gambar yang harus ditelaah satu per satu untuk mencari nodul yang dicurigai ganas, dan keputusan akhir mengenai jinak atau ganasnya suatu nodul sering kali baru bisa dipastikan lewat biopsi. Keterbatasan tenaga radiolog di banyak fasilitas kesehatan, ditambah risiko kelelahan visual pada pembacaan citra dalam jumlah besar, membuka ruang bagi sistem bantu diagnosis otomatis berbasis *Deep Learning* sebagai alat bantu skrining awal, bukan pengganti keputusan klinis.

*Convolutional Neural Network* (CNN) telah berulang kali terbukti efektif untuk tugas klasifikasi citra medis, termasuk citra CT paru-paru (Litjens et al., 2017; Shen et al., 2017). Pada klasifikasi citra CT paru-paru, model berbasis EfficientNet dilaporkan mencapai akurasi 99,10% pada dataset IQ-OTH/NCCD (*Iraq-Oncology Teaching Hospital/National Center for Cancer Diseases*) (Raza et al., 2023) dan mengungguli ResNet pada perbandingan langsung kedua arsitektur (Sandag & Kabo, 2024). Salah satu arsitektur CNN yang dipilih dalam penelitian ini adalah EfficientNet-B0, anggota terkecil dari keluarga EfficientNet yang diperkenalkan Tan & Le (2019). Keunggulan EfficientNet terletak pada strategi *compound scaling*, yaitu menaikkan kedalaman, lebar, dan resolusi jaringan secara serentak dengan rasio tetap, sehingga model mencapai akurasi tinggi tanpa membengkakkan jumlah parameter sebagaimana arsitektur CNN konvensional. Untuk mengatasi keterbatasan jumlah data medis berlabel, penelitian ini menerapkan *transfer learning*: bobot EfficientNet-B0 yang telah dilatih pada ImageNet dipakai sebagai titik awal, kemudian disesuaikan lewat *fine-tuning* pada citra CT paru-paru.

Tantangan yang justru lebih besar daripada pemilihan arsitektur adalah kualitas data itu sendiri. Sebagian besar dataset citra CT paru-paru yang beredar bebas di Kaggle merupakan kompilasi ulang dari sumber yang sama, disusun oleh pengunggah yang berbeda-beda tanpa *metadata* pasien yang jelas, sehingga rawan duplikasi lintas dataset maupun duplikasi di dalam dataset itu sendiri. Duplikasi semacam ini, bila tidak dideteksi sebelum data dibagi menjadi data latih dan data uji, menyebabkan *data leakage*: model seolah-olah mencapai akurasi tinggi karena menghafal citra identik yang pernah dilihat saat pelatihan, bukan karena benar-benar belajar mengenali pola nodul. Selain itu, sebagian dataset ini memberi label berdasarkan nama folder yang ditulis penyusun dataset, bukan berdasarkan diagnosis histopatologi yang dikonfirmasi laboratorium, sehingga ketepatan label itu sendiri patut dipertanyakan.

Untuk mengatasi celah tersebut, penelitian ini tidak berhenti pada satu sumber data. Selain mengumpulkan dan mengaudit tujuh dataset publik dari Kaggle, yang dideduplikasi lewat pencocokan *Message-Digest Algorithm* 5 (MD5) dan *perceptual hash* sebelum digabung, penelitian ini juga menambahkan LIDC-IDRI (*Lung Image Database Consortium and Image Database Resource Initiative*), koleksi data CT resmi dari *National Cancer Institute* yang didistribusikan lewat *The Cancer Imaging Archive* (TCIA) dan disertai anotasi nodul oleh hingga empat radiolog independen per kasus (Armato et al., 2011). Setiap nodul pada LIDC-IDRI diberi skor keganasan 1 sampai 5 oleh masing-masing radiolog, dan pada sebagian pasien tersedia pula data diagnosis histopatologi resmi dari TCIA yang menjadi rujukan koreksi label bila skor radiolog bersifat ambigu. Nodul dengan skor keganasan rata-rata tepat 3, yang berarti radiolog sendiri tidak sepakat apakah nodul itu jinak atau ganas, dilabeli ulang lewat pendekatan *nearest-neighbor* pada ruang fitur visual mengikuti metode yang diusulkan Zhang et al. (2022), bukan ditebak atau dibuang begitu saja.

Ketiga teknik optimasi yang tercermin pada judul penelitian ini, yaitu *fine-tuning*, *data augmentation*, dan *ensemble model*, diuji secara bertahap: mulai dari model tunggal, *ensemble* sederhana lewat *soft-voting* antara EfficientNet-B0 dan ResNet50 pada lima *fold*, hingga *ensemble stacking* yang memakai sebuah *meta-learner* untuk mempelajari cara terbaik menggabungkan keluaran kedua arsitektur tersebut alih-alih sekadar merata-ratakannya. Setiap tahap dibandingkan terhadap tahap sebelumnya berdasarkan akurasi, *cancer recall*, yaitu proporsi kasus kanker yang benar-benar tertangkap oleh model dan secara klinis lebih krusial daripada akurasi semata, serta *Receiver Operating Characteristic – Area Under the Curve* (ROC-AUC). Dengan cara ini, klaim performa akhir penelitian dapat ditelusuri ke eksperimen pembandingnya dan tidak berdiri sebagai angka tunggal yang asal-usulnya tidak diketahui.

## 1.2 Identifikasi Masalah

Berdasarkan latar belakang yang telah dijelaskan sebelumnya, persoalan penelitian ini tidak hanya terletak pada pemilihan arsitektur model, tetapi juga pada keabsahan data yang dipakai untuk melatih dan mengujinya, pada kontribusi nyata setiap teknik optimasi, serta pada cara hasilnya disajikan kepada pengguna. Masalah yang akan dicari solusinya dalam penelitian ini adalah sebagai berikut:

1. Bagaimana menerapkan *transfer learning* EfficientNet-B0 untuk mengklasifikasikan citra CT paru-paru ke dalam tiga kategori, yaitu *Malignant* (ganas), *Benign* (jinak), dan Normal?
2. Bagaimana mendeteksi dan menangani duplikasi citra, baik di dalam satu dataset maupun lintas dataset publik, serta memastikan bahwa data yang dipakai benar-benar citra CT, agar evaluasi model tidak bias akibat *data leakage* maupun citra yang salah jenis?
3. Bagaimana menangani nodul pada dataset LIDC-IDRI yang skor keganasannya ambigu karena tidak disepakati oleh radiolog, tanpa membuang data tersebut begitu saja?
4. Seberapa besar kontribusi masing-masing teknik optimasi, yaitu *fine-tuning*, *data augmentation*, dan *ensemble model*, terhadap peningkatan akurasi dan *cancer recall* dibandingkan model tanpa teknik tersebut?
5. Bagaimana membangun prototipe aplikasi *web* yang menjalankan konfigurasi model final dan menolak citra di luar domain CT paru-paru sebelum klasifikasi dijalankan?

## 1.3 Batasan Masalah

Dari identifikasi masalah tersebut, penelitian ini perlu dibatasi agar pembahasannya tetap terfokus pada optimasi *transfer learning* EfficientNet-B0 sebagaimana dinyatakan pada judul, dan agar hasilnya dapat diukur dengan sumber daya komputasi yang tersedia. Batasan masalah dalam penelitian ini adalah sebagai berikut:

1. Penelitian berfokus pada *transfer learning* dan *fine-tuning* atas model *pre-trained*, bukan melatih arsitektur CNN dari nol.
2. Arsitektur utama yang dievaluasi adalah EfficientNet-B0, dengan ResNet50 digunakan sebagai arsitektur kedua khusus untuk keperluan *ensemble*, bukan sebagai perbandingan independen yang setara.
3. Klasifikasi dibatasi pada tiga kelas pada level citra irisan CT dua dimensi, yaitu *Malignant*, *Benign*, dan Normal. Penelitian ini tidak melakukan segmentasi nodul maupun deteksi lokasi nodul pada citra.
4. Data yang digunakan seluruhnya berasal dari dataset publik yang telah tersedia secara daring, yaitu tujuh dataset dari Kaggle dan satu dataset dari TCIA (LIDC-IDRI), tanpa pengambilan data primer dari rumah sakit atau pasien secara langsung.
5. Pelatihan dibatasi maksimal 12 *epoch* untuk fase *feature extraction* dan 25 *epoch* untuk fase *fine-tuning* per model, dengan *early stopping* berdasarkan *macro-F1* validasi, mengingat keterbatasan sumber daya komputasi yang tersedia.
6. Evaluasi kinerja dibatasi pada metrik kuantitatif, yaitu akurasi, presisi, *recall* per kelas dengan penekanan pada *cancer recall*, *F1-Score*, dan ROC-AUC.
7. Penelitian ini menyertakan prototipe aplikasi *web* sederhana berbasis Streamlit untuk mendemonstrasikan hasil klasifikasi dan *ensemble* secara interaktif, dilengkapi lapisan validasi input yang menolak citra di luar domain, bukan sebagai sistem produksi yang siap dipakai di lingkungan klinis nyata.

## 1.4 Maksud dan Tujuan Penelitian

Maksud dari penelitian ini adalah mengimplementasikan dan mengoptimasi model *transfer learning* EfficientNet-B0 untuk klasifikasi kanker paru-paru pada citra CT, dengan data yang telah diaudit dan dikoreksi labelnya secara metodologis, lalu mengukur sumbangan *fine-tuning*, *data augmentation*, dan *ensemble model* terhadap performanya.

Tujuan yang ingin dicapai oleh penulis dari penelitian ini adalah:

1. Mengimplementasikan model *transfer learning* EfficientNet-B0 dengan skema dua fase, yaitu *feature extraction* lalu *fine-tuning*, untuk klasifikasi citra CT paru-paru ke dalam kelas *Malignant*, *Benign*, dan Normal.
2. Membangun jalur audit data yang mendeteksi duplikasi di dalam dan lintas dataset Kaggle, menyaring seri LIDC-IDRI berdasarkan jenis pemindaiannya, serta memproses berkas *Digital Imaging and Communications in Medicine* (DICOM) dan anotasi *Extensible Markup Language* (XML) radiolog menjadi label kelas yang dapat dipakai untuk pelatihan.
3. Menerapkan koreksi label pada nodul LIDC-IDRI yang ambigu, yaitu yang skor keganasan rata-ratanya tepat di titik tengah, melalui pendekatan *nearest-neighbor* pada ruang fitur visual serta melalui data diagnosis histopatologi resmi TCIA bila tersedia.
4. Mengukur kontribusi *fine-tuning*, *data augmentation*, dan *ensemble model* secara terpisah lewat perbandingan terkendali, lalu menentukan konfigurasi akhir dengan keseimbangan akurasi dan *cancer recall* terbaik.
5. Membangun prototipe aplikasi *web* yang menjalankan konfigurasi akhir tersebut, dilengkapi lapisan validasi input berbasis klasifikasi biner yang menolak citra di luar domain sebelum diproses model utama.

## 1.5 Manfaat Penelitian

Hasil penelitian ini diharapkan bermanfaat bagi pengembangan keilmuan di bidang klasifikasi citra medis maupun bagi pihak yang membutuhkan alat bantu skrining kanker paru-paru. Manfaat tersebut terbagi menjadi manfaat teoretis dan manfaat praktis sebagai berikut:

1. Manfaat Teoretis
   a. Memberikan kontribusi metodologis mengenai cara mengaudit dan menggabungkan dataset citra medis dari sumber yang heterogen, yaitu dataset Kaggle berbasis nama folder dan dataset akademik berbasis anotasi radiolog, tanpa mengorbankan validitas evaluasi akibat *data leakage*.
   b. Menjadi referensi bagi peneliti lain yang ingin menangani label ambigu pada dataset LIDC-IDRI tanpa membuang data tersebut, serta yang ingin mengukur kontribusi setiap teknik optimasi *transfer learning* secara terpisah.
2. Manfaat Praktis
   a. Menghasilkan model klasifikasi yang berpotensi membantu skrining awal nodul paru-paru pada citra CT, dengan penekanan pada *cancer recall* yang tinggi sebagai prioritas klinis.
   b. Menyediakan prototipe aplikasi *web* yang dapat dipakai sebagai alat peraga untuk menjelaskan cara kerja model klasifikasi citra medis kepada audiens non-teknis.

## 1.6 Metodologi Penelitian

Jenis penelitian yang digunakan adalah penelitian eksperimental, yaitu penelitian yang mengukur pengaruh suatu perlakuan dengan membandingkan hasil beberapa konfigurasi yang hanya berbeda pada satu faktor. Dalam penelitian ini, perlakuan tersebut adalah ketiga teknik optimasi pada judul, sedangkan hasilnya diukur dengan metrik evaluasi pada data uji yang terpisah dari data latih. Tahapan penelitian yang dilalui adalah sebagai berikut:

1. Studi literatur. Pada tahap ini penulis mengumpulkan referensi mengenai kanker paru-paru, interpretasi citra CT, arsitektur EfficientNet dan ResNet, *transfer learning*, serta metode penanganan label ambigu pada LIDC-IDRI (Zhang et al., 2022).
2. Pengumpulan dan audit data. Pada tahap ini penulis mengunduh tujuh dataset Kaggle dan dataset LIDC-IDRI beserta anotasi XML radiolognya dari TCIA. Setiap dataset Kaggle diaudit lewat pencocokan MD5 dan *perceptual hash* untuk mendeteksi duplikasi di dalam maupun lintas dataset, lalu labelnya dipetakan dari struktur folder ke tiga kelas target.
3. Pemrosesan data LIDC-IDRI. Pada tahap ini tag `Modality` pada kepala tiap berkas DICOM dibaca untuk memisahkan seri CT dari foto rontgen dada yang tersimpan berdampingan dalam struktur folder yang sama. Anotasi XML tiap radiolog kemudian di-*parsing*, penanda nodul yang saling berdekatan dikelompokkan menjadi nodul konsensus, dan label seri ditentukan dari skor keganasan rata-ratanya. Citra diekstraksi dari DICOM lewat *windowing Hounsfield Unit* (HU) pada irisan dada utuh 512×512 piksel, seragam dengan bentuk citra pada dataset Kaggle.
4. Koreksi label ambigu. Pada tahap ini nodul berlabel ambigu dilabeli ulang lewat pencarian *nearest-neighbor* pada ruang fitur EfficientNet-B0 *pre-trained* terhadap nodul yang labelnya sudah pasti, kemudian dikoreksi lebih lanjut memakai data diagnosis histopatologi resmi TCIA pada pasien yang datanya tersedia.
5. Penggabungan dataset. Pada tahap ini kumpulan data Kaggle dan LIDC-IDRI yang sudah berlabel final digabungkan, disertai pengecekan ulang duplikasi lintas kedua sumber, sebelum data dibagi memakai *Stratified Group K-Fold* dengan grup berdasarkan pasien atau kasus asal untuk mencegah kebocoran data.
6. Pelatihan model. Pada tahap ini EfficientNet-B0 dan ResNet50 dilatih pada lima *fold*, masing-masing melalui fase *feature extraction* lalu fase *fine-tuning*, dengan augmentasi data yang hanya diterapkan pada data latih.
7. Pengukuran kontribusi tiap teknik. Pada tahap ini sumbangan *fine-tuning*, *data augmentation*, dan *ensemble model* diukur secara terpisah lewat perbandingan terkendali: fase pertama melawan fase kedua untuk *fine-tuning*, pelatihan ulang dengan tiga kekuatan augmentasi untuk *data augmentation*, serta model tunggal melawan *soft-voting* dan *stacking* untuk *ensemble model*.
8. Pembangunan prototipe aplikasi. Pada tahap ini konfigurasi terbaik diimplementasikan ke dalam aplikasi Streamlit beserta lapisan validasi input, lalu diuji dengan *Black Box Testing*.
9. Penulisan laporan. Pada tahap ini penulis mendokumentasikan seluruh proses dan hasil penelitian ke dalam laporan skripsi, termasuk eksperimen yang hasilnya tidak sesuai harapan.

## 1.7 Sistematika Penulisan

Untuk memberi gambaran yang jelas tentang penelitian ini, disusunlah sistematika penulisan yang berisi materi yang dibahas pada setiap bab. Sistematika dalam penulisan skripsi ini adalah sebagai berikut:

**BAB I PENDAHULUAN**

Pada bab ini dijelaskan latar belakang dari topik penulisan skripsi, pokok permasalahan berupa identifikasi dan batasan masalah, maksud dan tujuan yang diharapkan dari penelitian, manfaat penelitian, metodologi yang digunakan, serta sistematika penulisan skripsi.

**BAB II TINJAUAN PUSTAKA**

Pada bab ini dijelaskan landasan teori yang berhubungan dengan penelitian, yaitu kanker paru-paru dan citra CT, *Deep Learning* dan CNN, *transfer learning*, arsitektur EfficientNet-B0 dan ResNet50, teknik optimisasi dan regularisasi, *data augmentation*, *ensemble model*, dataset LIDC-IDRI, perangkat lunak yang digunakan, penelitian terkait, serta metode evaluasi dan pengujian sistem.

**BAB III ANALISIS DAN PERANCANGAN**

Pada bab ini dijelaskan alur penelitian, analisis kebutuhan data beserta rincian audit dan pemrosesan tiap sumber dataset, kebutuhan perangkat keras dan perangkat lunak, perancangan skenario eksperimen, perancangan model, perancangan prototipe aplikasi, serta rancangan pengujian fungsionalitas aplikasi dengan *Black Box Testing*.

**BAB IV HASIL DAN PEMBAHASAN**

Pada bab ini dipaparkan hasil pengukuran kontribusi *fine-tuning*, *data augmentation*, dan *ensemble model* lewat perbandingan terkendali pada data uji yang sama, disertai penyesuaian ambang keputusan, analisis kesalahan klasifikasi, dan pemecahan performa menurut sumber data. Bab ini juga memuat implementasi model validasi biner dan prototipe aplikasi *web*, serta hasil *Black Box Testing* dan uji validasi input.

**BAB V KESIMPULAN DAN SARAN**

Bab ini merupakan penutup yang berisi kesimpulan sebagai jawaban atas identifikasi masalah pada Bab I, serta saran untuk pengembangan penelitian selanjutnya.
