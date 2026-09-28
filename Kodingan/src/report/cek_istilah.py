"""Cari kata berbahasa Inggris yang masih tercetak tegak di naskah.

Aturan template: kata dalam bahasa Inggris ditulis miring. Setiap kata tegak
(di luar kode, rumus, tautan, dan jalur gambar) dibandingkan frekuensinya pada
korpus bahasa Inggris dan bahasa Indonesia memakai pustaka wordfreq. Kata yang
umum dalam bahasa Inggris tetapi hampir tidak dipakai dalam bahasa Indonesia
dilaporkan beserta contoh kalimatnya, kecuali nama diri seperti nama perangkat
lunak, arsitektur, dataset, dan lembaga yang memang ditulis tegak.

Pemakaian: python cek_istilah.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

from wordfreq import zipf_frequency

sys.path.insert(0, str(Path(__file__).parent))
from build_docx import BAB, SRC, _tokens  # noqa: E402

import os
ENV_EN, ENV_SELISIH = os.environ.get("CEK_EN", "2.5"), os.environ.get("CEK_SELISIH", "1.5")
BERKAS = ["00_Halaman_Depan_dan_Abstrak.md"] + BAB + ["LAMPIRAN.md"]

# nama diri (perangkat lunak, model, dataset, lembaga, merek, fungsi kode) yang
# ditulis tegak, serta kata serapan yang lazim dipakai tegak dalam naskah
NAMA_DIRI = {
    "kaggle", "python", "pytorch", "torchvision", "streamlit", "numpy", "pandas", "scipy",
    "matplotlib", "pydicom", "imagehash", "hashlib", "scikit-learn", "imagenet", "lidc-idri",
    "tcia", "nvidia", "geforce", "rtx", "amd", "ryzen", "windows", "cuda", "adam", "adamw",
    "efficientnet", "efficientnet-b0", "resnet", "resnet50", "resnet101", "alexnet", "vgg16",
    "inception", "mobilenetv2", "inceptionv3", "efficientnetb0", "efficientnetb3", "efficientnetb1",
    "coco", "val2017", "github", "mcnemar", "lungct-net", "lung-effnet", "ver-net", "grad-cam",
    "shap", "javascript", "html", "apress", "springer", "normal", "dataset", "input", "data",
    "iq-oth", "nccd", "iq-othnccd", "unpad", "universitas", "padjadjaran", "sumedang",
    "jatinangor", "sep", "dr", "prof", "mt", "kom", "si", "phash", "md5", "dicom", "xml",
}


def kata_tegak(baris):
    """Kata-kata pada potongan yang tidak miring, bukan kode, dan bukan rumus."""
    for isi, b, it, kode in _tokens(baris):
        if it or kode:
            continue
        isi = re.sub(r"https?://\S+", " ", isi)
        for w in re.findall(r"[A-Za-z][A-Za-z\-']*[A-Za-z]|[A-Za-z]", isi):
            yield w


def main():
    temuan = defaultdict(list)
    for nama in BERKAS:
        teks = (SRC / nama).read_text(encoding="utf-8")
        kode = inggris = False
        for i, b in enumerate(teks.splitlines(), start=1):
            s = b.strip()
            if s.startswith("```"):
                kode = not kode
                continue
            if s.startswith("## ABSTRACT"):
                inggris = True
            elif s.startswith("## ") and inggris:
                inggris = False
            if kode or inggris or not s or s.startswith("|---") or s.startswith("!["):
                continue
            if s.startswith("***") and s.endswith("***"):
                continue  # judul berbahasa Inggris yang seluruhnya miring
            for w in kata_tegak(re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)):
                lw = w.lower().strip("-'")
                if len(lw) < 3 or lw in NAMA_DIRI:
                    continue
                en, idn = zipf_frequency(lw, "en"), zipf_frequency(lw, "id")
                if en >= float(ENV_EN) and en - idn >= float(ENV_SELISIH):
                    temuan[lw].append((nama[:6], i, s[:100]))
    for w, v in sorted(temuan.items(), key=lambda x: -len(x[1])):
        print(f"{w:22s} {len(v):3d}x  {v[0][0]}:{v[0][1]}  {v[0][2][:80]}")
    print(f"\n{len(temuan)} kata Inggris tegak ditemukan")
    return len(temuan)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
