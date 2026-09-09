# Duplikat Copy-Paste Windows (di dalam 1 dataset, 1 folder kelas)

Folder `contoh_01_folder_normal_kasus_10/` berisi ke-8 file yang ditemukan di
`Chest CT-Scan images Dataset (Mohamed Hany)/test/normal/` dengan nama:
`10.png`, `10 (2).png`, `10 - Copy.png`, `10 - Copy (2).png`, `10 - Copy (3).png`,
`10 - Copy - Copy.png`, `10 - Copy (2) - Copy.png`.

Pengecekan MD5 (lihat `Kodingan/outputs/manifests/full_manifest.csv`) menunjukkan kedelapan file
ini sebenarnya hanya **4 gambar unik**, masing-masing digandakan tepat 1 kali dengan pola
penamaan khas hasil *copy-paste* Windows Explorer ("- Copy", "- Copy (2)"):

| Gambar unik | Nama file kembar |
|---|---|
| A | `10 (2) - Copy.png` = `10 (2).png` |
| B | `10 - Copy (2) - Copy.png` = `10 - Copy (2).png` |
| C | `10 - Copy (3).png` = `10.png` |
| D | `10 - Copy - Copy.png` = `10 - Copy.png` |

Kemungkinan besar A, B, C, D adalah 4 slice CT berbeda dari studi/kasus #10 yang sama, yang
masing-masing kebetulan tergandakan satu kali saat proses pengumpulan data. Baik dedup MD5
maupun pengelompokan `case_key` (nomor kasus dari nama file) sama-sama menangkap pola ini --
lihat `Kejanggalan Dataset/03_duplikasi_dalam_satu_dataset.md`.
