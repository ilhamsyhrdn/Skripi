"""Periksa rujukan silang di naskah: setiap "Tabel x.y", "Gambar x.y", dan
"Bagian x.y(.z)" yang disebut di teks harus benar-benar ada.

Selain keberadaannya, dicetak pula kalimat di sekitar setiap rujukan tabel dan
gambar beserta judul sasarannya, supaya kecocokan isinya dapat diperiksa mata.
Keluaran berakhir dengan jumlah rujukan yang sasarannya tidak ditemukan.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

NASKAH = Path("D:/skripsi/Naskah Skripsi")
BAB = ["00_Halaman_Depan_dan_Abstrak.md", "BAB_I_Pendahuluan.md", "BAB_II_Tinjauan_Pustaka.md",
       "BAB_III_Analisis_dan_Perancangan.md", "BAB_IV_Hasil_dan_Pembahasan.md",
       "BAB_V_Kesimpulan_dan_Saran.md"]


def main(rinci=False):
    judul_tabel, judul_gambar, bagian = {}, {}, set()
    teks = {}
    for nama in BAB:
        baris = (NASKAH / nama).read_text(encoding="utf-8").splitlines()
        teks[nama] = baris
        for i, b in enumerate(baris):
            s = b.strip()
            m = re.match(r"^#{2,3}\s+(\d+\.\d+(?:\.\d+)?)\s", s)
            if m:
                bagian.add(m.group(1))
            m = re.match(r"^Tabel (\d+\.\d+) (.+)$", s)
            nxt = next((x.strip() for x in baris[i + 1:i + 3] if x.strip()), "")
            if m and nxt.startswith("|"):
                judul_tabel[m.group(1)] = m.group(2)
            m = re.match(r"^\*?Gambar (\d+\.\d+) (.+?)\*?$", s)
            prev = next((x.strip() for x in reversed(baris[:i]) if x.strip()), "")
            if m and prev.startswith("!["):
                judul_gambar[m.group(1)] = m.group(2)

    hilang = 0
    for nama, baris in teks.items():
        kode = False
        for i, b in enumerate(baris, start=1):
            s = b.strip()
            if s.startswith("```"):
                kode = not kode
            if kode or not s or s.startswith("![") or s.startswith("#"):
                continue
            for jenis, nomor in re.findall(r"\b(Tabel|Gambar|Bagian)\s+(\d+\.\d+(?:\.\d+)?)", s):
                # judul tabel/gambar itu sendiri bukan rujukan
                if re.match(rf"^\*?{jenis} {re.escape(nomor)} ", s) and jenis != "Bagian":
                    continue
                kamus = {"Tabel": judul_tabel, "Gambar": judul_gambar}.get(jenis)
                ada = (nomor in bagian) if jenis == "Bagian" else (nomor in kamus)
                if not ada:
                    hilang += 1
                    print(f"TIDAK ADA  {nama}:{i}  {jenis} {nomor}  <- {s[:90]}")
                elif rinci and jenis != "Bagian":
                    j = s.find(f"{jenis} {nomor}")
                    print(f"{nama[:10]}:{i:<4} {jenis} {nomor:5s} = {kamus[nomor][:55]:55s} | ...{s[max(0, j - 50):j + 30]}")
    print(f"\n{len(judul_tabel)} tabel, {len(judul_gambar)} gambar, {len(bagian)} bagian terdaftar; "
          f"{hilang} rujukan tanpa sasaran")
    sys.exit(1 if hilang else 0)


if __name__ == "__main__":
    main(rinci=len(sys.argv) > 1 and sys.argv[1] == "rinci")
