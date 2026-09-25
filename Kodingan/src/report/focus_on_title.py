"""Bring the manuscript back in line with its own title.

The title promises transfer learning EfficientNet-B0 optimised through
fine-tuning, data augmentation and ensemble modelling. Grad-CAM and the
input-validation (OOD) layer are neither named there nor required by it, so
they are removed from every chapter rather than carried as extra weight.

Bab IV is rewritten separately once the final experiment reports; this
script only handles the chapters whose edits do not depend on those numbers.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path("D:/skripsi/Naskah Skripsi")


def drop_lines(path: Path, predicates, renumber_prefix=None):
    """Remove whole lines matching any predicate; optionally renumber a list."""
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [ln for ln in lines if not any(p(ln) for p in predicates)]
    if renumber_prefix is not None:
        n = 0
        out = []
        for ln in kept:
            if re.match(rf"^\d+\.\s", ln) and renumber_prefix(ln):
                n += 1
                ln = re.sub(r"^\d+\.", f"{n}.", ln)
            out.append(ln)
        kept = out
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return len(lines) - len(kept)


def replace_in(path: Path, pairs):
    t = path.read_text(encoding="utf-8")
    n = 0
    for old, new in pairs:
        if old in t:
            t = t.replace(old, new)
            n += 1
        else:
            print(f"    [lewat] tidak ditemukan: {old[:60]}...")
    path.write_text(t, encoding="utf-8")
    return n


def drop_section(path: Path, heading_prefix: str):
    """Delete a '## <heading>' block up to the next '## ' heading."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out, skipping = [], False
    for ln in lines:
        if ln.startswith("## "):
            skipping = ln.startswith(heading_prefix)
        if not skipping:
            out.append(ln)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return len(lines) - len(out)


def main():
    # ---------------- Bab I ----------------
    p = SRC / "BAB_I_Pendahuluan.md"
    n = drop_lines(p, [
        lambda l: l.startswith("6. Bagaimana membangun mekanisme validasi input"),
        lambda l: l.startswith("5. Sistem dilengkapi model klasifikasi biner tambahan"),
        lambda l: l.startswith("6. Membangun mekanisme validasi input"),
    ])
    replace_in(p, [
        ("Penelitian ini menyertakan prototipe aplikasi web sederhana berbasis Streamlit "
         "untuk mendemonstrasikan hasil klasifikasi, ensemble, Grad-CAM, dan validasi input "
         "secara interaktif, bukan sebagai sistem produksi yang siap dipakai di lingkungan "
         "klinis nyata.",
         "Penelitian ini menyertakan prototipe aplikasi web sederhana berbasis Streamlit "
         "untuk mendemonstrasikan hasil klasifikasi dan ensemble secara interaktif, bukan "
         "sebagai sistem produksi yang siap dipakai di lingkungan klinis nyata."),
        ("7. Membangun prototipe aplikasi web yang mendemonstrasikan seluruh alur di atas, "
         "termasuk visualisasi Grad-CAM sebagai bentuk transparansi keputusan model.",
         "7. Membangun prototipe aplikasi web yang mendemonstrasikan seluruh alur di atas."),
        ("**Pembangunan Prototipe Aplikasi.** Mengimplementasikan model terbaik ke dalam "
         "aplikasi Streamlit yang mencakup prediksi, visualisasi Grad-CAM, dan validasi input.",
         "**Pembangunan Prototipe Aplikasi.** Mengimplementasikan model terbaik ke dalam "
         "aplikasi Streamlit untuk mendemonstrasikan hasil klasifikasi."),
    ])
    print(f"Bab I   : {n} baris dihapus")

    # ---------------- Bab II ----------------
    p = SRC / "BAB_II_Tinjauan_Pustaka.md"
    n = drop_section(p, "## 2.11 Grad-CAM")
    print(f"Bab II  : bagian Grad-CAM dihapus ({n} baris)")

    # ---------------- Daftar Pustaka ----------------
    p = SRC / "DAFTAR_PUSTAKA.md"
    n = drop_lines(p, [
        lambda l: l.startswith("Selvaraju, R. R."),
        lambda l: l.startswith("Lin, T.-Y., Maire, M."),
    ])
    print(f"Pustaka : {n} baris dihapus (Selvaraju/Grad-CAM, Lin/COCO)")

    print("\nBab III, Bab V, Lampiran, dan Bab IV ditangani terpisah "
          "karena perlu penyuntingan kalimat, bukan penghapusan baris utuh.")


if __name__ == "__main__":
    main()
