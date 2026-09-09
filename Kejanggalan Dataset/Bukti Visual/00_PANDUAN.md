# Panduan Bukti Visual

Folder ini berisi **file gambar CT asli** (disalin langsung dari `D:\skripsi\Dataset`, bukan
rekayasa) sebagai bukti konkret untuk setiap klaim di dokumen `Kejanggalan Dataset/*.md`. Setiap
subfolder di bawah ini punya `README.md` sendiri yang menjelaskan apa yang harus diperhatikan.

| Folder | Isinya | Dokumen terkait |
|---|---|---|
| `01_Leakage_Duplikat_Lintas_Dataset/` | Citra yang sama persis muncul di banyak dataset "berbeda" (2.017 salinan untuk 1 gambar!) | `01_duplikasi_lintas_dataset.md` |
| `02_Duplikat_Copy_Paste_Windows/` | 8 file "10*.png" yang ternyata hanya 4 gambar unik, digandakan lewat copy-paste Windows | `03_duplikasi_dalam_satu_dataset.md` |
| `03_Augmentasi_Sintetis_Menyamar_Data_Asli/` | Gambar dari dataset "Augmented" yang ternyata salinan mentah tanpa augmentasi apa pun | `02_dataset_augmentasi_sintetis.md` |
| `04_Data_Tanpa_Label_Terduplikasi/` | Gambar "Test cases" tanpa label yang diklaim 2 dataset sekaligus | `04_data_tanpa_label_dan_kualitas_gambar.md` |
| `05_Kegagalan_Deteksi_Otomatis_Hash_Collision/` | Bukti visual 2 pasien BERBEDA yang sempat salah tergabung karena tabrakan hash | `06_bug_metodologi_yang_ditemukan_dan_diperbaiki.md` |
| `06_Kandidat_Kemungkinan_Salah_Label_TIDAK_TERVERIFIKASI/` | **Baca README di dalamnya dulu** — hasil pencarian "gambar salah diagnosis" yang ternyata tidak terbukti, dengan bukti visual kenapa | (baru, khusus permintaan ini) |

## Ringkasan jujur untuk kategori "salah diagnosis"

Anda meminta bukti gambar yang "harusnya no-cancer tapi jadi cancer" (atau sebaliknya). Setelah
diperiksa dengan dua metode (audit ketat yang dipakai untuk keputusan pool kanonik, dan pencarian
tambahan yang lebih longgar khusus untuk permintaan ini):

- **Audit ketat (MD5 + perceptual-hash exact-match)**: 0 kasus ditemukan.
- **Pencarian longgar tambahan**: menemukan 405 "kandidat", tapi semuanya terbukti **false
  positive** setelah dicek visual langsung (lihat folder 06) — pola "1 gambar Normal cocok
  dengan 5 gambar Malignant berbeda sekaligus" adalah tanda tabrakan hash, bukan duplikat
  sungguhan, dan gambar-gambarnya memang terlihat jelas berbeda saat dibuka.

Kesimpulannya: **tidak ada bukti kuat gambar salah label** pada dataset yang dipakai untuk
training skripsi ini. Ini dilaporkan sebagai keterbatasan metode deteksi (bukan berarti mislabel
pasti tidak ada sama sekali di 7 dataset asli — hanya saja tidak bisa dideteksi secara andal
dengan metode hash pada domain citra CT paru-paru), sesuai catatan di dokumen 06 dan 08.
