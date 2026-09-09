# Augmentasi Sintetis (Das) yang Ternyata Salinan Asli Tanpa Augmentasi

Dataset "IQ-OTHNCCD Lung Cancer Dataset (Augmented) (Subhajeet Das)" mengklaim seluruh isinya
adalah hasil augmentasi. Audit menemukan **1.054 dari 3.609 file (~29%) byte-identical** (MD5
sama persis) dengan citra yang sudah ada di dataset lain -- artinya sebagian file di dalamnya
BUKAN hasil augmentasi baru, melainkan salinan mentah dari citra asli.

Setiap subfolder di sini berisi 2 file yang **identik secara MD5**:
- Satu dari `IQ-OTHNCCD Lung Cancer Dataset (Augmented) (Subhajeet Das)`
- Satu lagi dari `CT Scan Images for Lung Cancer (Dishan rathi20)` (yang sendiri adalah
  gabungan Mohamed Hany + IQ-OTH/NCCD, lihat dokumen 01)

Bandingkan kedua file di setiap subfolder -- keduanya akan terlihat identik. Lihat
`Kejanggalan Dataset/02_dataset_augmentasi_sintetis.md`.
