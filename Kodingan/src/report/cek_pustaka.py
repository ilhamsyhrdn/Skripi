"""Cocokkan kutipan di teks naskah dengan entri Daftar Pustaka.

Dua arah diperiksa: setiap kutipan (Penulis, tahun) di teks harus punya entri
di Daftar Pustaka, dan setiap entri Daftar Pustaka harus dikutip minimal sekali
di teks. Keluaran nol berarti keduanya konsisten.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

NASKAH = Path("D:/skripsi/Naskah Skripsi")
BAB = ["00_Halaman_Depan_dan_Abstrak.md", "BAB_I_Pendahuluan.md", "BAB_II_Tinjauan_Pustaka.md",
       "BAB_III_Analisis_dan_Perancangan.md", "BAB_IV_Hasil_dan_Pembahasan.md",
       "BAB_V_Kesimpulan_dan_Saran.md"]
NAMA = r"[A-Z][A-Za-z\-]+"
KUTIP = re.compile(rf"({NAMA})(?: et al\.| & ({NAMA}))?,? \(?((?:19|20)\d\d)\)?")


def kunci_entri(baris):
    """Kunci entri pustaka: (nama keluarga penulis pertama, tahun, jumlah penulis)."""
    m = re.match(r"^([^,(]+),.*?\(((?:19|20)\d\d)\)", baris)
    if not m:
        return None
    sebelum_tahun = baris[:baris.index(f"({m.group(2)})")]
    n = sebelum_tahun.count("&") + sebelum_tahun.count(".,") // 2 + 1
    return m.group(1).strip(), m.group(2), n


def main():
    entri = {}
    for b in (NASKAH / "DAFTAR_PUSTAKA.md").read_text(encoding="utf-8").splitlines():
        k = kunci_entri(b.strip())
        if k:
            entri[(k[0], k[1])] = b.strip()[:80]

    dikutip = {}
    for nama in BAB:
        t = (NASKAH / nama).read_text(encoding="utf-8")
        t = re.sub(r"```.*?```", "", t, flags=re.S)
        for m in KUTIP.finditer(t):
            penulis, tahun = m.group(1), m.group(3)
            if (penulis, tahun) in entri or re.search(rf"\b{penulis}\b.*?{tahun}", m.group(0)):
                dikutip.setdefault((penulis, tahun), set()).add(nama[:10])

    salah = 0
    hilang = sorted(k for k in dikutip if k not in entri)
    for k in hilang:
        salah += 1
        print(f"DIKUTIP TANPA ENTRI : {k[0]} ({k[1]}) di {sorted(dikutip[k])}")
    tak_dikutip = sorted(k for k in entri if k not in dikutip)
    for k in tak_dikutip:
        salah += 1
        print(f"ENTRI TAK DIKUTIP   : {k[0]} ({k[1]})")
    print(f"\n{len(entri)} entri pustaka, {len(dikutip)} sumber dikutip; {salah} ketidakcocokan")
    sys.exit(1 if salah else 0)


if __name__ == "__main__":
    main()
