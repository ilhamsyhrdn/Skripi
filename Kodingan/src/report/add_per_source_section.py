"""Insert the per-source performance analysis into Bab IV.

The headline 82,18% averages two populations of very different difficulty
(Kaggle 90,85% vs LIDC-IDRI 71,85%), which has to be stated plainly rather
than left for an examiner to discover. Adds one section and one table, then
renumbers everything that follows.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path("D:/skripsi/Naskah Skripsi/BAB_IV_Hasil_dan_Pembahasan.md")

SECTION_MAP = {"4.12": "4.13", "4.13": "4.14", "4.14": "4.15"}
TABLE_MAP = {"4.9": "4.10", "4.10": "4.11"}

NEW_SECTION = """
## 4.12 Analisis Performa per Sumber Data

Angka akhir 82,18% pada Tabel 4.7 merupakan performa terhadap keseluruhan 522 citra uji yang berasal dari dua sumber dengan karakteristik berbeda. Karena kedua sumber tidak memiliki tingkat kesulitan yang sama, performa model dipecah ulang menurut asal citranya agar angka tunggal tersebut tidak menutupi perbedaan yang ada. Hasilnya disajikan pada Tabel 4.9.

Tabel 4.9 Performa Konfigurasi Final Dipecah Menurut Sumber Data

| Sumber Data Uji | Jumlah Citra | Akurasi | Macro-F1 | Cancer Recall |
|---|---:|---:|---:|---:|
| Kaggle (irisan dada penuh) | 284 | 90,85% | 0,779 | 99,0% |
| LIDC-IDRI (crop nodul) | 238 | 71,85% | 0,623 | 80,3% |
| **Gabungan (angka yang dilaporkan)** | **522** | **82,18%** | **0,699** | **91,5%** |

Selisihnya besar: akurasi pada subset Kaggle 19 poin persentase lebih tinggi daripada pada subset LIDC-IDRI. Perbedaan ini konsisten di hampir semua kelas, sebagaimana dirinci pada Tabel 4.10.

Tabel 4.10 Recall per Kelas Dipecah Menurut Sumber Data

| Kelas | Kaggle (n) | Recall Kaggle | LIDC-IDRI (n) | Recall LIDC-IDRI |
|---|---:|---:|---:|---:|
| Malignant | 197 | 99,0% | 132 | 80,3% |
| Benign | 25 | 48,0% | 45 | 20,0% |
| Normal | 62 | 82,3% | 61 | 91,8% |

Penjelasan yang paling masuk akal atas pola ini terletak pada perbedaan cara kedua dataset dibentuk, bukan pada perbedaan cara citranya dipotong:

1. **Tingkat kesulitan kasus.** Label pada dataset Kaggle berasal dari nama folder yang disusun pengunggahnya, dan sebagian besar berisi kasus yang secara visual sudah jelas. Sebaliknya, LIDC-IDRI memuat nodul yang dianotasi langsung oleh radiolog beserta skor keganasan 1–5, termasuk 226 nodul yang skor rata-ratanya tepat di titik tengah karena para radiolog sendiri tidak sepakat. Kasus-kasus batas semacam itu memang secara inheren lebih sulit, dan kasus seperti ini justru tidak terwakili pada dataset Kaggle.

2. **Keragaman pasien.** Ketujuh dataset Kaggle pada dasarnya merupakan kompilasi ulang dari koleksi yang sama (terutama IQ-OTHNCCD), sehingga meskipun duplikasi persisnya sudah dibuang lewat audit, variasi pasien yang tersisa tetap lebih sempit dibandingkan LIDC-IDRI yang berasal dari 1.010 pasien berbeda.

3. **Bukan akibat pemotongan citra.** Dugaan bahwa model memakai perbedaan pembingkaian sebagai jalan pintas, yaitu menebak Benign hanya karena citranya berupa potongan (77,3% citra Benign memang berasal dari LIDC), tidak terbukti. Jika jalan pintas itu benar terjadi, recall Benign pada subset LIDC seharusnya justru tinggi; kenyataannya recall Benign pada LIDC hanya 20,0%, jauh di bawah subset Kaggle yang 48,0%. Dengan kata lain, model tidak mengambil untung dari pembingkaian, melainkan memang kesulitan pada data yang lebih sulit.

Implikasinya untuk pembacaan hasil penelitian ini: angka 82,18% sebaiknya dipahami sebagai performa pada populasi campuran, bukan sebagai performa yang akan diperoleh bila sistem dihadapkan pada data klinis nyata yang tingkat kesulitannya lebih menyerupai LIDC-IDRI. Untuk skenario semacam itu, angka 71,85% pada subset LIDC-IDRI merupakan estimasi yang lebih konservatif sekaligus lebih realistis. Penyajian kedua angka secara bersamaan dinilai lebih jujur daripada hanya melaporkan angka gabungan yang terlihat lebih baik.
"""


def renumber(text, mapping, pattern_tpl):
    tmp = text
    for old, new in mapping.items():
        tmp = re.sub(pattern_tpl.format(num=re.escape(old)),
                     lambda m, n=new: m.group(0).replace(old, f"@@{n}@@"), tmp)
    return tmp.replace("@@", "")


def main():
    t = SRC.read_text(encoding="utf-8")
    t = renumber(t, SECTION_MAP, r"## {num} ")
    t = renumber(t, TABLE_MAP, r"Tabel {num}\b")

    marker = "## 4.13 Visualisasi Grad-CAM"
    assert marker in t, "penanda Grad-CAM tidak ditemukan"
    t = t.replace(marker, NEW_SECTION.strip() + "\n\n" + marker)

    SRC.write_text(t, encoding="utf-8")
    print("bagian per-sumber ditambahkan")
    print("sub-bab :", re.findall(r"^## (4\.\d+)", t, flags=re.M))
    print("tabel   :", re.findall(r"^Tabel (4\.\d+)", t, flags=re.M))
    print("gambar  :", re.findall(r"^\*Gambar (4\.\d+)", t, flags=re.M))


if __name__ == "__main__":
    main()
