"""Miringkan istilah asing di naskah secara konsisten.

Aturannya mengikuti skripsi acuan yang diukur dari PDF-nya: istilah seperti
epoch, fold, loss, overfitting, transfer learning, web, dan optimizer hampir
selalu dicetak miring, sedangkan "input" dan "dataset" selalu tegak karena
dianggap kata serapan, begitu pula nama perangkat lunak seperti Streamlit dan
PyTorch.

Yang tidak disentuh: blok kode, kode sebaris, teks yang sudah miring, tautan,
jalur gambar, abstrak berbahasa Inggris, daftar pustaka, dan lampiran. Baris
yang tidak memuat istilah baru dibiarkan persis seperti aslinya.

Jalankan tanpa argumen untuk melihat rencana perubahan, dengan argumen "tulis"
untuk menerapkannya.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

NASKAH = Path("D:/skripsi/Naskah Skripsi")
BERKAS = ["00_Halaman_Depan_dan_Abstrak.md", "BAB_I_Pendahuluan.md", "BAB_II_Tinjauan_Pustaka.md",
          "BAB_III_Analisis_dan_Perancangan.md", "BAB_IV_Hasil_dan_Pembahasan.md",
          "BAB_V_Kesimpulan_dan_Saran.md"]

# frasa panjang lebih dulu agar "cancer recall" tidak terpecah menjadi "recall"
ISTILAH = sorted([
    "held-out test set", "black box testing", "transfer learning", "deep learning",
    "data augmentation", "feature extraction", "early stopping", "learning rate",
    "cross validation", "cancer recall", "benign recall", "cancer precision", "confusion matrix",
    "perceptual hash", "user interface", "file uploader", "ensemble stacking", "ensemble model",
    "computed tomography", "forward pass", "fine-tuning", "soft-voting", "meta-learner",
    "out-of-fold", "pre-trained", "nearest-neighbor", "overfitting", "underfitting", "epoch",
    "fold", "loss", "training", "recall", "ensemble", "stacking", "backbone", "threshold",
    "optimizer", "hyperparameter", "slider", "dropout", "scheduler", "pipeline", "checkpoint",
    "baseline", "batch", "softmax", "argmax", "framework", "debugging", "windowing", "default",
    "tensor", "logits", "web", "online", "gatekeeper", "thumbnail", "preprocessing",
    "script", "hyperparameter tuning",
], key=len, reverse=True)
POLA = re.compile(r"(?<![\w\-/.])(" + "|".join(re.escape(s) for s in ISTILAH) + r")(?![\w\-])",
                  re.IGNORECASE)


def potong(teks):
    """Pecah satu baris markdown menjadi potongan (isi, tebal, miring, kode)."""
    out, buf = [], []
    bold = italic = False
    i, n = 0, len(teks)

    def flush():
        if buf:
            out.append(["".join(buf), bold, italic, False]); buf.clear()

    while i < n:
        c = teks[i]
        if c == "`":
            j = teks.find("`", i + 1)
            if j > i:
                flush(); out.append([teks[i:j + 1], bold, italic, True]); i = j + 1; continue
        if teks.startswith("**", i):
            flush(); bold = not bold; i += 2; continue
        if c == "*" and (i + 1 < n and teks[i + 1] != " " or italic):
            flush(); italic = not italic; i += 1; continue
        buf.append(c); i += 1
    flush()
    return out


def rakit(potongan):
    """Susun ulang potongan menjadi markdown, memasang penanda saat gaya berubah."""
    hasil, b, it = [], False, False
    for isi, pb, pi, kode in potongan:
        if kode:
            if it and not pi:
                hasil.append("*"); it = False
            if b != pb:
                hasil.append("**"); b = pb
            hasil.append(isi); continue
        if it and not pi:
            hasil.append("*"); it = False
        if b != pb:
            hasil.append("**"); b = pb
        if pi and not it:
            hasil.append("*"); it = True
        hasil.append(isi)
    if it:
        hasil.append("*")
    if b:
        hasil.append("**")
    return "".join(hasil)


def miringkan_baris(baris):
    ptg = potong(baris)
    baru, berubah = [], False
    for isi, b, it, kode in ptg:
        if kode or it or not POLA.search(isi):
            baru.append([isi, b, it, kode]); continue
        pos = 0
        for m in POLA.finditer(isi):
            if m.start() > pos:
                baru.append([isi[pos:m.start()], b, False, False])
            baru.append([m.group(0), b, True, False])
            pos = m.end(); berubah = True
        if pos < len(isi):
            baru.append([isi[pos:], b, False, False])
    return (rakit(baru), True) if berubah else (baris, False)


def olah(teks):
    baris = teks.split("\n")
    kode = inggris = False
    ubah = 0
    for k, b in enumerate(baris):
        s = b.strip()
        if s.startswith("```"):
            kode = not kode; continue
        if s.startswith("## ABSTRACT"):
            inggris = True
        if kode or inggris or not s:
            continue
        if s.startswith("|---") or re.match(r"^!\[.*\]\(.*\)$", s):
            continue
        # Keterangan gambar pada skripsi acuan dicetak tegak, jadi pembungkus
        # miring *Gambar ...* dilepas lebih dulu, lalu istilah asing di dalamnya
        # dimiringkan satu per satu seperti teks biasa.
        m = re.match(r"^\*(Gambar \d+\.\d+ .*)\*$", s)
        if m:
            isi, _ = miringkan_baris(m.group(1))
            baris[k] = isi; ubah += 1
            continue
        if s.startswith("#"):
            awal = b[:len(b) - len(b.lstrip("#"))] + " "
            isi, ok = miringkan_baris(b[len(awal):])
        elif s.startswith("|"):
            sel = b.strip().strip("|").split("|")
            olahan = [miringkan_baris(c) for c in sel]
            ok = any(o for _, o in olahan)
            isi = "|" + "|".join(t for t, _ in olahan) + "|"
            awal = ""
        else:
            awal, isi0 = "", b
            isi, ok = miringkan_baris(isi0)
        if ok:
            baris[k] = awal + isi; ubah += 1
    return "\n".join(baris), ubah


def main():
    tulis = len(sys.argv) > 1 and sys.argv[1] == "tulis"
    total = 0
    for nama in BERKAS:
        p = NASKAH / nama
        raw = p.read_text(encoding="utf-8")
        nl = "\r\n" if "\r\n" in raw else "\n"
        baru, n = olah(raw.replace("\r\n", "\n"))
        total += n
        print(f"{nama:40s} {n:3d} baris berubah")
        if tulis and n:
            p.write_text(baru.replace("\n", nl), encoding="utf-8")
    print(f"\n{'DITULIS' if tulis else 'RENCANA'}: {total} baris")


if __name__ == "__main__":
    main()
