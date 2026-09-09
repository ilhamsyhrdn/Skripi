# Kandidat Kemungkinan Salah Label -- HASIL: TIDAK TERBUKTI (Penting Dibaca)

Sesuai permintaan, dilakukan pencarian tambahan khusus untuk kasus "gambar yang harusnya
no-cancer tapi terlabel cancer (atau sebaliknya)" -- dengan kriteria yang lebih longgar dari
metode utama (perceptual-hash jarak <=4 DAN average-hash jarak <=4 DAN resolusi identik),
dibatasi hanya pada pasangan lintas label (Normal vs Malignant, Normal vs Benign).

**Hasil: audit utama (MD5 exact + perceptual-hash exact) yang dipakai untuk keputusan pool
kanonik menemukan 0 (nol) kasus gambar identik dengan label bertentangan** -- lihat
`audit_summary.json`, field `label_conflict_groups: 0`. Ini kabar baik: tidak ada bukti
mislabel yang solid pada data yang dipakai training.

Pencarian tambahan yang lebih longgar (khusus untuk permintaan ini) memang menemukan 405
pasangan "kandidat", TETAPI setelah diperiksa visual secara langsung, pasangan-pasangan ini
adalah **false positive** dari masalah yang sama seperti di folder 05 (citra CT paru-paru
punya struktur visual yang mirip secara umum -- FOV melingkar, kontras serupa -- sehingga
hash tidak cukup diskriminatif).

Contoh di `contoh_01_fan_out_palsu/`: `Normal case (8).jpg` "cocok" dengan **5 gambar
Malignant berbeda sekaligus** (`Malignant case (89/90/100/101/102).jpg`) pada pencarian
longgar ini. Satu gambar tidak mungkin secara valid identik dengan 5 gambar lain yang
berbeda-beda -- pola "satu lawan banyak" ini adalah tanda klasik tabrakan hash, bukan
duplikat sungguhan. **Buka dan bandingkan sendiri gambar-gambar di folder ini** -- akan
terlihat jelas keduanya adalah pasien yang berbeda (bentuk tubuh, pola pembuluh darah paru,
dan anatomi tulang rusuk berbeda).

**Kesimpulan jujur untuk skripsi:** tidak ditemukan bukti kuat adanya citra yang salah
label/salah diagnosis pada dataset yang dipakai. Kalaupun ada mislabel yang sesungguhnya
tersembunyi di salah satu dari 7 dataset asli, metode yang tersedia (hash-based) tidak
mampu mendeteksinya secara andal pada domain citra CT paru-paru -- ini didokumentasikan
sebagai keterbatasan (limitation) penelitian, bukan diklaim sebagai temuan positif.
