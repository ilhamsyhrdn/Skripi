# BAB V

# KESIMPULAN DAN SARAN

Bab ini merangkum kesimpulan dari seluruh rangkaian penelitian yang telah dilaksanakan, serta menyampaikan saran bagi pengembangan penelitian selanjutnya.

## 5.1 Kesimpulan

Berdasarkan hasil penelitian, implementasi, dan pengujian yang telah dilakukan, dapat ditarik kesimpulan sebagai berikut:

1. Implementasi *transfer learning* EfficientNet-B0 dua fase (*feature extraction* diikuti *fine-tuning*) berhasil dilakukan untuk klasifikasi kanker paru-paru pada citra CT ke dalam tiga kelas: Malignant, Benign, dan Normal, dengan konfigurasi *hyperparameter* (Adam, *ReduceLROnPlateau*, *early stopping* berbasis macro-F1) yang stabil di seluruh lima *fold* pengujian.

2. Audit duplikasi lintas dan di dalam tujuh dataset Kaggle berhasil mendeteksi dan mengeluarkan citra augmentasi sintetis serta salinan duplikat yang berpotensi menyebabkan *data leakage*, menghasilkan pool data Kaggle final sebanyak 1.627 citra yang bersih dari duplikasi.

3. Pemrosesan dataset LIDC-IDRI dari berkas DICOM dan anotasi XML radiolog berhasil menghasilkan 1.308 citra berlabel, termasuk penanganan 226 nodul dengan skor keganasan ambigu lewat pendekatan *nearest-neighbor* pada ruang fitur visual (mengikuti Zhang et al., 2022), sehingga tidak ada data yang harus dibuang akibat ambiguitas label radiolog. Koreksi lanjutan memakai diagnosis histopatologi resmi TCIA, yang mencakup 157 dari 1.010 pasien dan beririsan dengan 130 pasien pada pool ini, mengubah label 142 dari 227 seri pemindaian yang tersentuh, yang menegaskan bahwa label berbasis skor subjektif radiolog semata tidak selalu identik dengan diagnosis klinis sebenarnya.

4. Sebanyak 197 citra Kaggle yang semula tidak berlabel diperiksa terlebih dahulu terhadap seluruh citra berlabel lewat pencocokan MD5 dan *perceptual hash*, dan tidak satu pun cocok, sehingga label aslinya memang tidak dapat dipulihkan. Label 166 di antaranya kemudian diperkirakan lewat *k-nearest-neighbor* dengan syarat kesepakatan empat dari lima tetangga. Karena distribusi hasilnya sangat miring (154 Malignant, 12 Normal, tanpa Benign) dan kemungkinan hanya mencerminkan proporsi kelas data rujukannya, seluruh citra tersebut dibatasi hanya boleh masuk data latih dan dilarang muncul pada data uji.

5. *Fine-tuning* memberikan kontribusi yang konsisten meski tidak besar: macro-F1 validasi rata-rata naik dari 0,6607 pada fase *feature extraction* menjadi 0,6841 setelah *fine-tuning*, dan sembilan dari sepuluh model membaik. Manfaatnya tidak merata antar arsitektur, yaitu +0,0401 pada ResNet50 tetapi hanya +0,0067 pada EfficientNet-B0. Pengukuran langsung menunjukkan penyebabnya bukan porsi jaringan yang dibuka, melainkan jumlah bobot yang tersedia untuk disesuaikan: tiga blok terakhir mencakup 23,3 juta parameter pada ResNet50 tetapi hanya 3,2 juta pada EfficientNet-B0 yang memang dirancang hemat parameter.

6. *Data augmentation* memberikan kontribusi yang berbeda sifatnya dari dua teknik lainnya. Pengujian atas tiga konfigurasi, yaitu tanpa augmentasi, augmentasi ringan yang disesuaikan sifat citra CT, dan augmentasi penuh, menunjukkan bahwa ketiganya tidak dapat dibedakan secara statistik pada data uji; uji McNemar untuk ketiga pasangan menghasilkan nilai p sebesar 0,80, 0,27, dan 0,61, seluruhnya jauh di atas 0,05. Sebaran akurasi antar sepuluh *fold* di dalam satu konfigurasi (simpangan baku 2,88 hingga 3,17 poin) bahkan sekitar tiga kali lebih besar daripada selisih terbesar antar konfigurasi (1,03 poin). Sebaliknya, pada pengendalian *overfitting* pengaruhnya sangat jelas dan teratur: jurang antara akurasi latih dan validasi turun dari 10,01 poin tanpa augmentasi menjadi 5,34 poin pada augmentasi ringan dan 0,70 poin pada augmentasi penuh. Dengan demikian augmentasi terbukti menjalankan fungsinya sebagai penahan hafalan data latih, tetapi pada data penelitian ini penahanan tersebut tidak berbuah peningkatan performa yang terukur, karena *overfitting* memang bukan faktor pembatas utama setelah *early stopping*, penyimpanan bobot terbaik, dan penggabungan *ensemble* diterapkan.

7. *Ensemble model* memberikan kontribusi terbesar di antara ketiga teknik yang diuji. *Soft-voting* menaikkan akurasi dari rata-rata 74,99% menjadi 77,51%, dan *ensemble stacking* dengan *meta-learner* regresi logistik pada prediksi *out-of-fold* menaikkannya lebih jauh menjadi 81,60% dengan *cancer recall* 90,35%, tanpa satu pun model dasar dilatih ulang.

8. Kenaikan akibat *stacking* tidak sepenuhnya gratis dan perlu dinyatakan terbuka. Macro-F1-nya (0,682) justru sedikit di bawah *soft-voting* (0,717), karena *meta-learner* yang dilatih memaksimalkan kecocokan keseluruhan cenderung menekan kelas Benign yang hanya menyumbang 13,1% data latih. Peningkatan akurasi dan *cancer recall* karena itu sebagian dibayar dengan performa pada kelas minoritas.

9. Konfigurasi final (*ensemble stacking*, ambang 0,50) mencapai akurasi 81,60%, macro-F1 0,682, *cancer recall* 90,35%, presisi Malignant 88,36%, dan ROC-AUC Malignant 0,945 pada 489 citra uji yang sama sekali tidak dilibatkan dalam pelatihan maupun pemilihan model manapun. Ambang 0,50 dipilih meski ambang 0,35 memberikan akurasi sedikit lebih tinggi (82,41%), karena keunggulan tersebut diperoleh dengan menekan kelas minoritas lebih jauh lagi.

10. Perbandingan ROC-AUC dan *recall* pada kelas Benign menunjukkan bahwa kesulitan model pada kelas ini lebih bersifat persoalan kalibrasi daripada ketidakmampuan mengenali pola. ROC-AUC Benign mencapai 0,855, yang berarti model mampu memberi skor relatif lebih tinggi pada citra Benign, tetapi *recall*-nya hanya 25,0% karena pada keputusan *argmax* kelas lain hampir selalu memenangkan probabilitas.

11. Analisis kesalahan menunjukkan model umumnya keliru ketika sedang ragu, bukan keliru dengan penuh keyakinan. Pada rentang keyakinan di atas 0,90, yang mencakup 45,4% data uji, model benar pada 99,5% kasus, sementara 74,4% seluruh kesalahan terkumpul pada rentang keyakinan 0,40 hingga 0,75. Pola ini dapat dimanfaatkan secara praktis sebagai penanda kapan hasil prediksi perlu ditinjau ulang secara manual.

12. Pemecahan hasil menurut sumber data mengungkap bahwa angka 81,60% merupakan rata-rata dari dua populasi dengan tingkat kesulitan yang jauh berbeda, yaitu 96,61% pada subset Kaggle dan 58,76% pada subset LIDC-IDRI. Perbedaan 37,9 poin persentase ini bersumber dari karakteristik datanya: LIDC-IDRI memuat nodul beranotasi radiolog berukuran beberapa milimeter pada irisan dada utuh, termasuk 226 kasus batas yang para radiolognya sendiri tidak sepakat, sementara dataset Kaggle didominasi kasus yang secara visual sudah jelas. Untuk memperkirakan performa pada data klinis nyata yang kesulitannya lebih menyerupai LIDC-IDRI, angka 58,76% merupakan estimasi yang jauh lebih konservatif dan lebih realistis daripada angka gabungan.

13. Prototipe aplikasi web berbasis Streamlit berhasil dibangun dan mendemonstrasikan seluruh alur klasifikasi, mulai dari pengunggahan citra hingga penyajian probabilitas tiap anggota ensemble. Kesembilan skenario *Black Box Testing* pada aplikasi tersebut dinyatakan berhasil.

14. Lapisan validasi input berbasis klasifikasi biner berhasil diterapkan sebagai gerbang yang menyaring unggahan sebelum model klasifikasi utama dijalankan. Pengujian pada 20 citra, terdiri dari 12 citra CT paru-paru dan 8 citra objek umum dari dataset COCO, memutuskan seluruhnya dengan benar. Jarak antara kedua kelompok sangat lebar, yaitu keyakinan terendah 85,0% pada citra CT yang diterima berbanding keyakinan tertinggi 6,6% pada citra COCO yang ditolak. Hasil sempurna ini perlu dibaca proporsional karena kedua kelompok memang sangat kontras secara visual, dan pengujian terhadap kasus batas yang lebih mirip seperti CT organ lain atau citra MRI belum dilakukan.

## 5.2 Saran

Berdasarkan keterbatasan dan temuan dalam penelitian ini, berikut saran untuk pengembangan penelitian selanjutnya:

1. **Perbaikan Recall Kelas Benign.** Kelas Benign secara konsisten menjadi kelas paling sulit di seluruh konfigurasi, dan Kesimpulan nomor 10 menunjukkan persoalannya terletak pada ambang keputusan, bukan pada ketidakmampuan model mengenali pola. Penelitian selanjutnya dapat mengeksplorasi teknik yang menyasar persoalan tersebut secara langsung, misalnya *focal loss*, kalibrasi probabilitas per kelas, ambang keputusan khusus kelas Benign, atau pengumpulan data tambahan khusus kelas ini.

2. **Meta-Learner yang Sadar Ketidakseimbangan Kelas.** *Meta-learner* pada penelitian ini adalah regresi logistik biasa yang memaksimalkan kecocokan keseluruhan, sehingga menekan kelas minoritas sebagaimana dibahas pada Kesimpulan nomor 8. Melatih *meta-learner* dengan bobot kelas, atau memakai *meta-learner* lain seperti *gradient boosting* atau jaringan saraf kecil, berpotensi mempertahankan kenaikan akurasi tanpa mengorbankan macro-F1.

3. **Pembatasan Area Perhatian Model (Lung Masking).** Perbedaan performa yang besar antara subset Kaggle dan LIDC-IDRI menunjukkan model kesulitan ketika nodul hanya menempati bagian kecil dari irisan utuh. Penerapan segmentasi paru sebagai pra-pemrosesan, yaitu menutup seluruh area di luar parenkim paru sebelum citra masuk ke model, berpotensi memusatkan kapasitas model pada area yang benar-benar relevan.

4. **Pengujian Resolusi Masukan yang Lebih Tinggi.** Penelitian ini memakai resolusi 512×512 piksel karena itulah ukuran asli citra pada kedua sumber. Bila tersedia data DICOM dengan resolusi lebih tinggi, atau perangkat keras yang memungkinkan pemotongan beresolusi penuh di sekitar nodul, pengaruh resolusi terhadap kemampuan model mengenali nodul kecil dapat diuji lebih jauh.

5. **Pengujian Lapisan Validasi Input pada Kasus Batas.** Lapisan validasi input saat ini hanya diuji membedakan citra CT paru-paru dari citra objek umum, yang secara visual sangat kontras. Pengujian lanjutan sebaiknya menyertakan citra medis lain yang jauh lebih mirip, misalnya CT abdomen, CT kepala, foto rontgen dada, atau citra MRI, untuk mengetahui apakah gerbang ini tetap andal pada kasus yang lebih sulit.

6. **Validasi Data Klinis Independen.** Model ini dilatih dan diuji sepenuhnya pada data publik. Validasi eksternal memakai data CT dari rumah sakit mitra yang belum pernah dilihat sama sekali oleh proses pengembangan model ini penting untuk menguji ketahanan model terhadap variasi alat pemindai dan protokol akuisisi di lapangan.

7. **Penambahan Arsitektur Ketiga pada Ensemble.** Menambahkan arsitektur yang berbeda secara struktural, misalnya *Vision Transformer*, pada *ensemble stacking* berpotensi menambah keragaman prediksi yang digabungkan, yang dalam penelitian ini terbukti menjadi sumber peningkatan performa terbesar.

8. **Segmentasi Nodul.** Penelitian ini dibatasi pada klasifikasi level citra tanpa segmentasi lokasi nodul. Penambahan modul segmentasi (misalnya U-Net) dapat melengkapi sistem ini dengan kemampuan menunjukkan batas nodul secara presisi.

9. **Pembaruan Berkelanjutan pada Koreksi Label.** Basis data diagnosis histopatologi TCIA yang dipakai penelitian ini hanya mencakup 157 dari 1.010 pasien LIDC-IDRI. Bila data diagnosis tambahan untuk pasien lain tersedia di masa depan, koreksi label dapat diperluas untuk mencakup lebih banyak kasus, dan label perkiraan pada 166 citra Kaggle dapat digantikan label sebenarnya bila sumber aslinya ditemukan.
