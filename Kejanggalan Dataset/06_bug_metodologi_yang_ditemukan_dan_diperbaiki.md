# Kejanggalan #6 — Bug Metodologi yang Ditemukan & Diperbaiki Saat Membangun Alat Audit Ini

Dokumen ini didokumentasikan secara sengaja (mengikuti gaya pencatatan yang sama dengan proyek
sebelumnya di repository ini, lih. commit `01837e2`) supaya proses audit dataset ini transparan dan
dapat diperiksa ulang — termasuk kesalahan yang sempat terjadi di tengah proses pembuatannya sendiri.

## Percobaan 1 — deteksi near-duplicate lintas sumber dengan threshold longgar

Percobaan pertama menyatukan gambar lintas dataset yang jarak Hamming perceptual-hash (phash)
64-bit-nya ≤ 6 bit. Hasilnya: **80.000 lebih pasangan tergabung**, dan pool kanonik kelas Benign
runtuh dari seharusnya ratusan gambar mentah menjadi hanya **24 grup** — angka yang tidak masuk akal
mengingat sumber data sendiri mengklaim ±120 pasien Benign unik.

**Penyebab:** citra CT paru-paru adalah domain visual dengan entropi rendah — semua irisan aksial
memiliki bentuk FOV melingkar yang sama, kontras jendela paru yang serupa, dan tata letak tulang
rusuk/tulang belakang yang mirip antar pasien manapun. Perceptual hash 64-bit (berbasis DCT frekuensi
rendah) ternyata **tidak cukup diskriminatif** untuk domain ini — pasien yang benar-benar berbeda pun
bisa memiliki jarak hash yang sangat kecil murni karena kemiripan struktur global, bukan karena
gambar yang sama.

## Percobaan 2 — mengganti "nomor kasus berdekatan" dengan "nomor kasus identik", tapi masih tercampur dataset sintetis

Percobaan kedua mengganti heuristik "nomor kasus berdekatan + hash longgar" dengan aturan yang lebih
ketat: hanya menyatukan file dengan bukti nama-file yang identik (`case_key` sama persis). Namun,
dataset augmentasi sintetis (Subhajeet Das) masih ikut serta dalam proses pencocokan hash exact.
Hasilnya: satu grup gabungan tunggal berisi **5.629 gambar** yang mencampur label Normal dan Benign
sekaligus, membuat seluruh grup itu ditandai `label_conflict=True` dan **seluruh kelas Benign hilang
total (0 gambar)** dari pool kanonik.

**Penyebab (diverifikasi langsung, bukan dugaan):** dua gambar hasil augmentasi dari dataset Das —
`Benign cases/vertical_flip (14).jpg` dan `Normal cases/vertical_flip (96).jpg` — kebetulan memiliki
perceptual-hash 64-bit yang **identik persis**, padahal MD5 keduanya berbeda (isi piksel sungguh
berbeda). Ini adalah **tabrakan hash murni (hash collision)**, bukan duplikat sungguhan. Karena kedua
gambar "sintetis" ini masing-masing juga cocok (exact-hash) dengan gambar-gambar riil di dataset lain
melalui rantai transitif *union-find*, satu tabrakan hash yang salah ini menjalar dan menggabungkan
ribuan gambar riil dari label berbeda menjadi satu grup besar yang saling bertentangan.

## Perbaikan final

1. Dataset augmentasi sintetis (Das) **dikeluarkan total dari proses union-find/pengelompokan grup**
   sejak awal — bukan cuma dari hasil akhir. Dataset ini tetap diperiksa, tapi hanya secara terpisah
   dan searah (dicocokkan ke pool data riil untuk statistik bukti overlap), tidak pernah menjadi
   bagian dari rantai penggabungan grup.
2. Pencocokan near-duplicate lintas sumber yang longgar (jarak phash) **tidak lagi dipakai untuk
   menggabungkan grup secara otomatis** — hanya dipakai untuk menghasilkan daftar kandidat
   (`cross_source_near_duplicate_candidates.csv`, mensyaratkan dua hash independen — phash **dan**
   ahash — setuju) yang dilaporkan sebagai bukti pendukung, bukan keputusan otomatis.
3. Penggabungan grup otomatis hanya dipercayakan pada tiga sinyal yang berbasis **bukti kuat**, bukan
   kemiripan visual semata: (a) MD5 identik persis (byte-for-byte), (b) perceptual-hash identik
   persis (64-bit sama persis, bukan mendekati), dan (c) bukti nama-file identik (`case_key` sama)
   dalam folder sumber yang sama.

Hasil akhir setelah perbaikan: 0 grup dengan label bertentangan, dan distribusi kelas kanonik
(Malignant 665 / Normal 136 / Benign 93) yang jauh lebih masuk akal dibanding kedua percobaan
sebelumnya. Lihat `05_ketidakseimbangan_kelas.md` untuk pembahasan hasil akhir ini.

## Pelajaran untuk metodologi skripsi

Temuan ini secara langsung mendukung pilihan metodologi bab 3 proposal skripsi: keputusan deduplikasi
otomatis untuk data medis **tidak boleh hanya mengandalkan kemiripan visual/perceptual similarity**
tanpa verifikasi tambahan, karena domain citra medis (khususnya CT dengan struktur anatomi yang
serupa antar pasien) memiliki risiko *false positive* yang jauh lebih tinggi dibanding foto natural
pada umumnya — sebuah nuansa yang tidak dibahas eksplisit pada tinjauan pustaka data augmentation
umum (Chlap et al., 2021) karena artikel tersebut membahas augmentasi untuk training, bukan
deduplikasi untuk audit dataset.
