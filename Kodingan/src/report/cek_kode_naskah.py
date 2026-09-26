"""Periksa bahwa setiap cuplikan kode di naskah benar-benar ada di kode sumber.

Cuplikan kode pada skripsi wajib disalin dari berkas yang sungguh dijalankan,
bukan ditulis ulang dari ingatan. Pemeriksaan ini mencocokkan tiap baris kode
di naskah (setelah spasi di awal dan akhirnya dibuang) dengan baris-baris kode
sumber. Baris penanda bagian yang dilewati, yaitu komentar berisi "...",
diabaikan karena memang bukan salinan.

Keluaran nol berarti semua cuplikan cocok; selain itu setiap baris yang tidak
ditemukan dicetak beserta nomor baris naskahnya.
"""
from __future__ import annotations

import sys
from pathlib import Path

NASKAH = Path("D:/skripsi/Naskah Skripsi")
KODE = Path("D:/skripsi/Kodingan")


def baris_sumber() -> set[str]:
    kumpulan = set()
    for f in list(KODE.glob("src/**/*.py")) + list(KODE.glob("streamlit_app/**/*.py")):
        if "__pycache__" in f.parts:
            continue
        for b in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            if b.strip():
                kumpulan.add(b.strip())
    return kumpulan


def main():
    sumber = baris_sumber()
    gagal = 0
    for md in sorted(NASKAH.glob("BAB_*.md")):
        baris = md.read_text(encoding="utf-8").splitlines()
        dalam, mulai = False, 0
        for i, b in enumerate(baris, start=1):
            if b.strip().startswith("```"):
                dalam, mulai = (not dalam), i
                continue
            if not dalam:
                continue
            s = b.strip()
            if not s or ("..." in s and s.startswith("#")) or s == "...":
                continue
            if s not in sumber:
                gagal += 1
                print(f"{md.name}:{i} (blok mulai baris {mulai}) tidak ditemukan di sumber: {s[:90]}")
    print(f"\n{'SEMUA COCOK' if gagal == 0 else f'{gagal} baris tidak cocok'}")
    sys.exit(1 if gagal else 0)


if __name__ == "__main__":
    main()
