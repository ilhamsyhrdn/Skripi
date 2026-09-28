"""Bangun DAFTAR_GAMBAR.md dan DAFTAR_TABEL.md langsung dari isi naskah.

Kedua daftar ini sebelumnya ditulis manual, sehingga mudah tertinggal ketika
judul gambar atau tabel di dalam bab berubah. Dengan membacanya langsung dari
berkas bab, keduanya dijamin selalu sama dengan isi naskah.

Keterangan gambar dikenali dari baris `Gambar x.y ...` tepat di bawah baris
gambarnya, sedangkan judul tabel dikenali dari baris `Tabel x.y ...` yang
berdiri sendiri di atas tabelnya. Rujukan di tengah kalimat tidak ikut terbaca
karena tidak berada di awal baris.
"""
from __future__ import annotations

import re
from pathlib import Path

NASKAH = Path("D:/skripsi/Naskah Skripsi")
BAB = ["BAB_I_Pendahuluan.md", "BAB_II_Tinjauan_Pustaka.md",
       "BAB_III_Analisis_dan_Perancangan.md", "BAB_IV_Hasil_dan_Pembahasan.md",
       "BAB_V_Kesimpulan_dan_Saran.md"]
OUT_GAMBAR = NASKAH / "DAFTAR_GAMBAR.md"
OUT_TABEL = NASKAH / "DAFTAR_TABEL.md"

POLA_GAMBAR = re.compile(r"^(Gambar \d+\.\d+ .+?)\s*$")
POLA_TABEL = re.compile(r"^(Tabel \d+\.\d+ .+?)\s*$")


def urutan(judul: str) -> tuple[int, int]:
    bab, nomor = re.search(r"(\d+)\.(\d+)", judul).groups()
    return int(bab), int(nomor)


def kumpulkan():
    gambar, tabel = {}, {}
    for nama in BAB:
        f = NASKAH / nama
        if not f.exists():
            print(f"  [lewat] {nama} tidak ditemukan")
            continue
        baris = f.read_text(encoding="utf-8").splitlines()
        for i, mentah in enumerate(baris):
            b = mentah.strip()
            m = POLA_GAMBAR.match(b)
            sebelumnya = next((x.strip() for x in reversed(baris[:i]) if x.strip()), "")
            if m and sebelumnya.startswith("!["):
                gambar.setdefault(m.group(1).split()[1], m.group(1))
                continue
            m = POLA_TABEL.match(b)
            if not m or "|" in b:
                continue
            # Judul tabel yang sebenarnya selalu diikuti baris tabel dalam dua
            # baris berikutnya, sedangkan kalimat rujukan diikuti prosa.
            lanjutan = [x.strip() for x in baris[i + 1:i + 3]]
            if any(x.startswith("|") for x in lanjutan):
                tabel.setdefault(m.group(1).split()[1], m.group(1))
    return gambar, tabel


def tulis(path: Path, judul: str, isi: list[str]):
    baris = [f"# {judul}", ""] + [b for j in isi for b in (j, "")]
    path.write_text("\n".join(baris).rstrip() + "\n", encoding="utf-8")


def main():
    gambar, tabel = kumpulkan()
    g = [gambar[k] for k in sorted(gambar, key=urutan)]
    t = [tabel[k] for k in sorted(tabel, key=urutan)]
    tulis(OUT_GAMBAR, "DAFTAR GAMBAR", g)
    tulis(OUT_TABEL, "DAFTAR TABEL", t)
    print(f"{len(g)} gambar ke {OUT_GAMBAR.name} dan {len(t)} tabel ke {OUT_TABEL.name}", flush=True)


if __name__ == "__main__":
    main()
