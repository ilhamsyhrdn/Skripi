# Kegagalan Deteksi Otomatis: Tabrakan Hash Palsu (BUKAN Bukti Salah Diagnosis)

**Penting: folder ini BUKAN contoh gambar yang salah label/salah diagnosis.** Ini adalah bukti
visual dari sebuah *false positive* yang sempat terjadi saat mengembangkan alat audit ini sendiri.

`vertical_flip (14).jpg` (folder "Benign cases") dan `vertical_flip (96).jpg` (folder "Normal
cases") -- keduanya dari dataset augmentasi sintetis Subhajeet Das -- memiliki **perceptual hash
64-bit yang identik persis**, padahal MD5-nya berbeda (isi piksel sungguh berbeda; keduanya
adalah hasil flip vertikal dari 2 pasien yang berbeda). Buka dan bandingkan kedua file di folder
ini secara visual -- Anda akan melihat keduanya memang gambar yang berbeda, bukan gambar yang
sama dengan label tertukar.

Tabrakan hash inilah yang sempat (sebelum diperbaiki) menyebabkan seluruh kelas Benign hilang
dari pool kanonik karena tergabung secara keliru dengan kelas Normal melalui rantai transitif.
Kronologi lengkap ada di `Kejanggalan Dataset/06_bug_metodologi_yang_ditemukan_dan_diperbaiki.md`.
