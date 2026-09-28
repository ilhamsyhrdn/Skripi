"""Periksa aturan template: uraian setiap subbab dan sub-subbab minimal 2 paragraf,
dan setiap paragraf minimal 5 baris.

Jumlah baris dihitung dengan meniru pemenggalan baris Word: kata demi kata
ditempatkan pada baris selebar 14 cm (baris pertama 13 cm karena inden 1 cm),
dengan lebar tiap kata diukur dari berkas huruf Times New Roman 12 pt sesuai
tebal atau miringnya. Yang dihitung sebagai paragraf hanya paragraf uraian; butir
daftar, tabel, keterangan gambar, persamaan, kode, dan keterangan nama kode
tidak ikut dihitung.

Subbab yang masih memiliki sub-subbab (subbab induk) cukup berisi paragraf
pengantar, mengikuti contoh pada template, sehingga yang diperiksa ketat hanya
subbab dan sub-subbab paling bawah.

Pemakaian: python cek_paragraf.py [nama_berkas_bab ...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_docx import BAB, SRC, _lebar_cm, _tokens  # noqa: E402

LEBAR = 14.0
INDEN = 1.0
MIN_BARIS = 5
MIN_PARAGRAF = 2


def jumlah_baris(teks, inden=True):
    """Tiru pemenggalan baris Word pada paragraf rata kiri-kanan 12 pt."""
    kata = []
    for isi, b, it, kode in _tokens(teks.replace(" -- ", " – ")):
        for w in re.split(r"(\s+)", isi):
            if w and not w.isspace():
                lebar = _lebar_cm(w, 10.5 if kode else 12, b, it)
                kata.append(lebar)
    spasi = _lebar_cm(" ", 12)
    baris, posisi, awal = 1, (INDEN if inden else 0.0), True
    for w in kata:
        perlu = w if awal else spasi + w
        if posisi + perlu > LEBAR + 1e-6 and not awal:
            baris, posisi = baris + 1, w
        else:
            posisi += perlu
        awal = False
    return baris


def bagian(path: Path):
    """Kumpulkan paragraf uraian per subbab/sub-subbab."""
    baris = path.read_text(encoding="utf-8").splitlines()
    hasil, cur, dalam_kode = [], None, False
    for b in baris:
        s = b.strip()
        if s.startswith("```"):
            dalam_kode = not dalam_kode
            continue
        if dalam_kode or not s:
            continue
        m = re.match(r"^(#{2,3})\s+(.*)$", s)
        if m:
            cur = {"judul": m.group(2), "level": len(m.group(1)), "paragraf": []}
            hasil.append(cur)
            continue
        if cur is None or s.startswith(("#", "|", "![", "$$", "---")):
            continue
        if re.match(r"^(\d+|[a-z])\.\s", s) or re.match(r"^[-*]\s", s):
            continue
        if re.match(r"^\*\*[^*].*\*\*$", s) and "**" not in s[2:-2].replace("***", ""):
            continue
        if re.match(r"^\*?(Gambar|Tabel) \d+\.\d+ ", s) or s.startswith("*Script*/kode program"):
            continue
        cur["paragraf"].append(s)
    for k, sec in enumerate(hasil):
        nxt = hasil[k + 1] if k + 1 < len(hasil) else None
        sec["induk"] = nxt is not None and nxt["level"] > sec["level"]
    return hasil


# Subbab berbentuk daftar pada Bab I dan Bab V (identifikasi masalah, batasan,
# tujuan, manfaat, metodologi, sistematika, kesimpulan, saran) mengikuti contoh
# template: satu paragraf pengantar yang diikuti butir-butir bernomor.
SUBBAB_DAFTAR = {"1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "5.1", "5.2"}


def main(berkas):
    total = 0
    for nama in berkas:
        for sec in bagian(SRC / nama):
            nomor = sec["judul"].split()[0]
            daftar = nomor in SUBBAB_DAFTAR
            ukuran = [jumlah_baris(p) for p in sec["paragraf"]]
            # di subbab berbentuk daftar, kalimat pengantar butir (berakhir titik
            # dua) dan uraian di bawah label bab pada sistematika boleh pendek
            pendek = [(u, p[:60]) for u, p in zip(ukuran, sec["paragraf"])
                      if u < MIN_BARIS and not (daftar and (p.endswith(":") or nomor == "1.7"))]
            kurang = (not sec["induk"]) and (not daftar) and len(ukuran) < MIN_PARAGRAF
            if kurang or pendek:
                total += 1
                tanda = "INDUK " if sec["induk"] else ""
                print(f"{nama[:7]} {tanda}{sec['judul'][:55]:55s} paragraf={len(ukuran)} baris={ukuran}")
                for u, p in pendek:
                    print(f"      {u} baris: {p}")
    print(f"\n{total} subbab belum memenuhi aturan 2 paragraf x 5 baris")
    return total


if __name__ == "__main__":
    args = sys.argv[1:] or BAB
    sys.exit(1 if main(args) else 0)
