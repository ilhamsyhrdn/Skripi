# Ringkasan Eksekutif — Audit Kejanggalan Dataset

**Skripsi:** Optimasi Transfer Learning EfficientNet-B0 untuk Klasifikasi Kanker Paru-Paru pada Citra CT
**Tanggal audit:** 2026-09-09
**Alat audit:** `Kodingan/src/audit/build_manifest.py` dan `Kodingan/src/audit/make_splits.py`
**Sumber data mentah:** `D:\skripsi\Dataset\` (7 dataset CT paru-paru yang diunduh terpisah dari Kaggle)

## Apa yang ditemukan

Ketujuh folder di `Dataset/` **tidak berisi 7 sumber data yang independen**. Setelah dilakukan audit
berbasis hash (MD5 exact-match + perceptual hash + bukti nama file) terhadap seluruh 12.882 file
gambar, ditemukan bahwa mayoritas file adalah **duplikat atau near-duplikat lintas dataset** dari
hanya dua "keluarga" data asli:

1. **IQ-OTH/NCCD** (Al-Yasriy et al.) — diunggah ulang identik oleh 3 uploader Kaggle berbeda,
   ditambah 1 versi augmentasi sintetis dari uploader ke-4.
2. **Kaggle "Chest CT-Scan images" (mohamedhanyyy)** — tertanam utuh (≈85% file identik secara byte)
   di dalam 2 dataset lain yang diberi nama seolah-olah dataset baru.

| Tahap | Jumlah gambar |
|---|---:|
| Total file gambar di `Dataset/` (7 folder) | 12.882 |
| File korup/tidak terbaca | 0 |
| Dikecualikan: dataset augmentasi sintetis (Subhajeet Das) | 3.609 |
| Dikecualikan: folder "Test cases" tanpa label | 394 |
| Sisa berlabel & bukan sintetis, sebelum dedup | 8.879 |
| **Setelah deduplikasi (MD5 + phash exact + bukti nama file)** | **894 gambar unik** |

Artinya **≈93% dari seluruh file yang ada di folder `Dataset/` adalah salinan/near-duplikat**, bukan
data pasien baru. Bila hal ini tidak dideteksi dan dataset digabung lalu displit secara acak seperti
biasa, model apa pun — sekuat apa pun arsitekturnya — akan mendapat *data leakage* yang membuat
akurasi validasi/tes tampak sangat tinggi secara palsu (persis seperti yang pernah terjadi pada
proyek sebelumnya di repository ini, lihat commit `01837e2` dan `623189f`).

## Keputusan yang diambil (dan mengapa)

1. **Hanya 894 gambar unik (deduplikasi ketat) yang dipakai untuk train/val/test**, bukan 12.882.
   Lihat `01_duplikasi_lintas_dataset.md` dan `03_duplikasi_dalam_satu_dataset.md`.
2. **Dataset augmentasi sintetis (Subhajeet Das) dibuang total** dari pool kanonik — bukan cuma dari
   split, tapi dari SELURUH proses deduplikasi, karena awalnya justru merusak proses deduplikasi itu
   sendiri. Lihat `02_dataset_augmentasi_sintetis.md` dan `06_bug_metodologi_yang_ditemukan_dan_diperbaiki.md`.
3. **Folder "Test cases" (394 file) dibuang** karena tidak berlabel ground-truth. Lihat
   `04_data_tanpa_label_dan_kualitas_gambar.md`.
4. **Split train/val/test dilakukan pada level "grup" (bukan level file gambar)**, dengan grup
   ditentukan dari kombinasi hasil deduplikasi + bukti nomor kasus pada nama file, supaya slice CT
   yang berdekatan dari satu studi/pasien yang sama tidak pernah terpisah antara train dan test.
   Lihat `07_strategi_split.md`.
5. Kelas final sangat tidak seimbang (Malignant 665 : Normal 136 : Benign 93, rasio ≈7,15:1) — lihat
   `05_ketidakseimbangan_kelas.md`. Ini yang mendasari penggunaan *class-weighted loss*,
   *fine-tuning* bertahap, dan terutama *ensemble model* pada metodologi skripsi ini: dengan data
   bersih yang terbatas, mengandalkan satu model tunggal jauh lebih rentan varians tinggi
   dibandingkan menggabungkan beberapa model.
6. **Dampak leakage dibuktikan secara kuantitatif** (bukan cuma diklaim): baseline yang sengaja
   dibuat bocor (split acak per gambar, tanpa dedup) mencapai akurasi 94,5% / macro-F1 0,881,
   dibanding pipeline final yang jujur di 85,5% / 0,695 — selisih +9,0 poin akurasi yang **murni
   palsu** akibat 99,85% gambar "test"-nya sudah punya salinan di "train". Lihat
   `08_bukti_kuantitatif_dampak_leakage.md`.

## Bukti visual (file gambar asli, bukan cuma angka)

Folder `Bukti Visual/` berisi **salinan file gambar asli** untuk setiap kategori kejanggalan di
atas — bisa langsung dibuka dan dibandingkan. Termasuk hasil pencarian jujur untuk kategori
"gambar salah diagnosis/salah label": ditemukan 405 kandidat pada pencarian longgar, tapi
**semuanya terbukti false-positive** setelah dicek visual (lihat
`Bukti Visual/06_Kandidat_Kemungkinan_Salah_Label_TIDAK_TERVERIFIKASI/`). Mulai dari
`Bukti Visual/00_PANDUAN.md`.

## Berkas hasil audit

Semua bukti mentah (bukan cuma klaim) tersimpan di `Kodingan/outputs/manifests/`:

- `full_manifest.csv` — setiap file gambar + hash + grup + flag.
- `duplicate_groups.csv` — semua grup duplikat (>1 anggota), diurutkan dari yang terbesar.
- `cross_source_near_duplicate_candidates.csv` — kandidat near-duplikat lintas sumber (dua hash
  independen setuju), dilaporkan tapi **tidak** digabung otomatis (lihat alasan di dokumen 06).
- `canonical_pool.csv` — 894 gambar unik final yang dipakai untuk training.
- `trainval_folds.csv`, `test_holdout.csv`, `split_summary.json` — hasil split leakage-safe.
- `audit_summary.json` — ringkasan angka lengkap dari seluruh proses audit.
