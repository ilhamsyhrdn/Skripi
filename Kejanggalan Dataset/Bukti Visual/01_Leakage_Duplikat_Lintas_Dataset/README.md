# Leakage / Duplikat Lintas Dataset

Setiap subfolder berisi **citra CT yang secara pixel identik atau sama persis**, tetapi
disimpan dengan nama file berbeda di dataset Kaggle yang berbeda-beda -- seolah-olah data baru,
padahal isinya sama.

## contoh_01_normal_case_1_muncul_di_6_dataset/
Satu citra "Normal case (1)" dari studi IQ-OTH/NCCD asli, ditemukan **identik** (MD5/perceptual
hash sama) di **6 dari 7 dataset** yang diunduh:
- `The_IQ_OTHNCCD_lung_cancer_dataset____Hamdalla_F.__Al_Yasriy_)__Normal case (1).jpg` (sumber asli)
- `IQ_OTHNCCD___Lung_Cancer_Dataset__Aditya_Mahimkar___Normal case (1).jpg`
- `CT_Scan_Images_of_Lung_Cancer_Patients._MD._NAFEES_IMTIAZ___Normal cases (1).jpg`
- `Lung_cancer_dataset__IQ_OTHNCCD__Waseim_Nagah_Hennes___003828_02_01_174.png`
- `Chest_CT_Scan_images_Dataset__Mohamed_Hany___10 (2) - Copy.png`
- `CT_Scan_Images_for_Lung_Cancer__Dishan_rathi20___10 (2) - Copy.png`

Total anggota grup ini di seluruh dataset: **2.017 salinan** dari 1 citra yang sama (lihat
`Kodingan/outputs/manifests/duplicate_groups.csv`, `group_id=53`).

## contoh_02_adenocarcinoma_muncul_di_4_dataset/
Satu citra adenocarcinoma yang sama muncul di 4 dataset berbeda dengan 4 skema penamaan berbeda
(`000050 (3).png`, `adenocarcinoma (178).png`, `0000051.png`, dst.) -- bukti bahwa dataset
"Mohamed Hany", "Dishan Rathi", "Nafees Imtiaz", dan "Hennes" bersumber dari data yang sama.

**Kalau dibiarkan dan digabung lalu displit acak per gambar**, salinan-salinan ini bisa saja
terpisah -- satu di train, satu di test -- membuat model seolah-olah "mengenali" gambar test
padahal sudah pernah melihat salinan pixel-identiknya saat training. Lihat
`Kejanggalan Dataset/01_duplikasi_lintas_dataset.md` untuk penjelasan lengkap.
