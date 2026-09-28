"""Bangun naskah skripsi (.docx) dari berkas-berkas markdown di folder Naskah Skripsi.

Format disalin dari template resmi program studi, yaitu berkas
"Templete Penyusunan  SKRIPSI.docx", beserta aturan pada komentar-komentarnya:

- A4, margin atas 4 cm, kiri 4 cm, bawah 3 cm, kanan 3 cm; header dan footer 1,25 cm.
- Times New Roman 12 pt; isi rata kiri-kanan, 2 spasi, inden baris pertama 1 cm.
- Judul bagian awal dan judul bab tebal, kapital, di tengah, dan berjarak 4 spasi
  ke isinya.
- Subbab dan sub-subbab tebal dengan nomor tanpa titik di akhir, berjarak 12 pt
  dari uraian subbab sebelumnya.
- Judul tabel di atas tabel dan keterangan gambar di bawah gambar dengan huruf
  standar; isi tabel 1 spasi; kepala tabel berlatar C5E0B3 dan diulang di setiap
  halaman.
- Persamaan bernomor per bab, misalnya (2.1), di sisi kanan.
- Kode program berada di dalam kotak, 10 pt, 1 spasi, dan diawali keterangan
  nama kodenya.
- Nomor halaman romawi di tengah bawah pada bagian awal; angka di tengah bawah
  pada halaman pertama bab dan di kanan atas pada halaman berikutnya.

Daftar isi, daftar tabel, daftar gambar, dan daftar lampiran ditulis sebagai field
Word, lalu diisi nomor halamannya oleh update_fields_word.ps1.

Pemakaian:
    python build_docx.py          naskah lengkap  -> Skripsi_Lengkap.docx
    python build_docx.py bab1     sampai Bab I    -> Skripsi_Bab_I.docx
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Length, Pt, RGBColor

SRC = Path("D:/skripsi/Naskah Skripsi")
FIG = Path("D:/skripsi/Kodingan/outputs/figures/final")
LOGO = SRC / "Gambar" / "Logo_Unpad.jpeg"
MML2OMML = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")

BODY_FONT = "Times New Roman"
MONO_FONT = "Consolas"         # kode program: 9 s.d. 11 pt, lebih kecil dari isi
MONO_EM = 0.5498               # lebar satu karakter Consolas (1126/2048 em)
BODY_SIZE = Pt(12)
CODE_SIZE = Pt(9)
TABLE_SIZES = (Pt(12), Pt(11), Pt(10))
INDENT = Cm(1.0)
TEXT_WIDTH_CM = 14.0           # 21 cm - 4 cm - 3 cm
HEADER_FILL = "C5E0B3"         # warna kepala tabel pada template
BARIS = 13.8                   # tinggi satu baris spasi tunggal Times New Roman 12 pt
JARAK_SUBBAB = Pt(12)

BAB = ["BAB_I_Pendahuluan.md", "BAB_II_Tinjauan_Pustaka.md",
       "BAB_III_Analisis_dan_Perancangan.md", "BAB_IV_Hasil_dan_Pembahasan.md",
       "BAB_V_Kesimpulan_dan_Saran.md"]

# jalur gambar di markdown -> (berkas gambar final, lebar cm); lebar tidak boleh
# melebihi lebar teks 14 cm
FIGURES = {
    "Gambar/Gambar_2.1_Arsitektur_CNN.png": ("gambar_2_1_arsitektur_cnn.png", 14.0),
    "Gambar/Gambar_2.2_Blok_MBConv_EfficientNet.png": ("gambar_2_2_blok_mbconv.png", 14.0),
    "Gambar/Gambar_2.3_Blok_Residual_ResNet50.png": ("gambar_2_3_blok_residual.png", 14.0),
    "Gambar/Gambar_3.1_Diagram_Alur_Penelitian.png": ("gambar_3_1_alur_penelitian.png", 14.0),
    "Gambar/Gambar_3.2_Sampel_Kaggle_Al-Yasriy.png": ("gambar_3_2_sampel_kaggle_alyasriy.png", 9.5),
    "Gambar/Gambar_3.3_Sampel_Kaggle_Rathi.png": ("gambar_3_3_sampel_kaggle_rathi.png", 9.5),
    "Gambar/Gambar_3.4_Sampel_LIDC.png": ("gambar_3_4_sampel_lidc.png", 9.5),
    "Gambar/Gambar_3.5_Use_Case_Diagram.png": ("gambar_3_5_use_case.png", 14.0),
    "Gambar/Gambar_4.1_Confusion_Matrix_Stacking.png": ("gambar_4_5_confusion_matrix.png", 10.0),
    "Gambar/Gambar_4.2_Kurva_ROC_Stacking.png": ("gambar_4_6_kurva_roc.png", 11.0),
    "Gambar/Gambar_4.3_Confusion_Matrix_Validasi_Biner.png": ("gambar_4_cm_validasi_biner.png", 9.0),
    "Gambar/Gambar_4.4_Aplikasi_Beranda.png": ("gambar_4_9_aplikasi_beranda.png", 14.0),
    "Gambar/Gambar_4.5_Aplikasi_Prediksi.png": ("gambar_4_10_aplikasi_prediksi.png", 14.0),
    "Gambar/Gambar_4.6_Aplikasi_Tolak.png": ("gambar_4_11_aplikasi_tolak.png", 14.0),
}

# keadaan penomoran selama penyusunan: nomor bab (untuk persamaan), urutan
# persamaan, dan apakah paragraf berikutnya tepat berada di bawah judul
STATE = {"bab": 0, "eq": 0, "bawah_judul": False, "tanpa_inden": False}
ROMAWI = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}


# ------------------------------------------------------------------ utilitas
def _font(run, size=BODY_SIZE, bold=None, italic=None, name=BODY_FONT, underline=None):
    run.font.name = name
    rf = run._element.get_or_add_rPr().get_or_add_rFonts()
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), name)
    run.font.size = size
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if underline is not None:
        run.underline = underline
    return run


def _field(run, instr_text):
    """Sisipkan field sederhana (PAGE, TOC, dan sejenisnya) ke dalam run."""
    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
    instr.text = instr_text
    sep = OxmlElement("w:fldChar"); sep.set(qn("w:fldCharType"), "separate")
    txt = OxmlElement("w:t"); txt.text = " "
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    for el in (begin, instr, sep, txt, end):
        run._r.append(el)


def _pf(p, line=2.0, before=0, after=0, indent=None, align=None, keep=None):
    pf = p.paragraph_format
    pf.line_spacing = line
    pf.space_before = before if isinstance(before, Length) else Pt(before)
    pf.space_after = after if isinstance(after, Length) else Pt(after)
    pf.first_line_indent = Cm(0) if indent is None else indent
    if align is not None:
        p.alignment = align
    if keep is not None:
        pf.keep_with_next = keep
    return pf


# ------------------------------------------------------------------- dokumen
def _gaya(st, size=BODY_SIZE, bold=False, italic=False):
    """Samakan gaya ke Times New Roman hitam; atribut huruf dan warna tema dibuang
    karena di OOXML atribut tema mengalahkan nama huruf dan warna yang ditulis."""
    st.font.name = BODY_FONT
    st.font.size = size
    st.font.bold = bold
    st.font.italic = italic
    st.font.color.rgb = RGBColor(0, 0, 0)
    rpr = st.element.get_or_add_rPr()
    rf = rpr.get_or_add_rFonts()
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), BODY_FONT)
    color = rpr.find(qn("w:color"))
    if color is not None:
        for a in ("w:themeColor", "w:themeShade", "w:themeTint"):
            if color.get(qn(a)) is not None:
                del color.attrib[qn(a)]


def _gaya_baru(doc, name, base="Normal"):
    try:
        return doc.styles[name]
    except KeyError:
        st = doc.styles.add_style(name, 1)
        st.base_style = doc.styles[base]
        st.quick_style = False
        return st


def setup_document():
    doc = Document()
    normal = doc.styles["Normal"]
    _gaya(normal)
    pf = normal.paragraph_format
    pf.space_before = Pt(0); pf.space_after = Pt(0); pf.line_spacing = 2.0

    # Gaya judul bawaan dipakai agar field daftar isi dapat membacanya.
    for name, align in (("Heading 1", WD_ALIGN_PARAGRAPH.CENTER),
                        ("Heading 2", WD_ALIGN_PARAGRAPH.LEFT),
                        ("Heading 3", WD_ALIGN_PARAGRAPH.LEFT)):
        st = doc.styles[name]
        _gaya(st, bold=True)
        st.paragraph_format.alignment = align
        st.paragraph_format.line_spacing = 2.0
        st.paragraph_format.space_before = Pt(0)
        st.paragraph_format.space_after = Pt(0)
        st.paragraph_format.first_line_indent = Cm(0)
        st.paragraph_format.keep_with_next = True

    # judul yang tampil seperti judul bab tetapi tidak masuk daftar isi
    st = _gaya_baru(doc, "Judul Tanpa Daftar")
    _gaya(st, bold=True)
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # entri daftar isi mengikuti gaya TOC template: tab kanan bertitik di tepi
    # kanan teks, nomor subbab di 0 cm dan judulnya di 1 cm
    kanan = Cm(TEXT_WIDTH_CM)
    for lvl, (line, before, after, left, hang) in {
            1: (1.5, 6, 0, 0.0, 0.0),
            2: (1.15, 0, 5, 1.0, 1.0),
            3: (1.15, 0, 5, 2.25, 1.25),
            9: (1.5, 0, 0, 2.0, 2.0)}.items():
        # gaya bawaan Word bernama huruf kecil ("toc 1"); gaya "TOC 1" buatan
        # sendiri tidak akan dipakai oleh field daftar isi
        st = _gaya_baru(doc, f"toc {lvl}")
        # jadikan definisi gaya bawaan Word (styleId TOC1, bukan gaya kustom);
        # bila ditandai kustom, Word menamainya ulang dan memakai gayanya sendiri
        st.element.set(qn("w:styleId"), f"TOC{lvl}")
        if st.element.get(qn("w:customStyle")) is not None:
            del st.element.attrib[qn("w:customStyle")]
        _gaya(st)
        f = st.paragraph_format
        f.line_spacing = line
        f.space_before = Pt(before); f.space_after = Pt(after)
        f.left_indent = Cm(left); f.first_line_indent = Cm(-hang)
        f.right_indent = Cm(0.8)
        if hang:
            f.tab_stops.add_tab_stop(Cm(left))
        f.tab_stops.add_tab_stop(kanan, WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)

    # keterangan gambar dan judul tabel: huruf standar, di tengah, gaya tersendiri
    # agar daftar gambar dan daftar tabel terbentuk terpisah
    for name in ("Caption Gambar", "Caption Tabel"):
        st = _gaya_baru(doc, name)
        _gaya(st)
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        st.paragraph_format.line_spacing = 2.0
        st.paragraph_format.first_line_indent = Cm(0)

    st = _gaya_baru(doc, "Judul Lampiran")
    _gaya(st, bold=True)
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    st.paragraph_format.first_line_indent = Cm(0)
    st.paragraph_format.keep_with_next = True

    # bingkai halaman tidak ikut mengelilingi header dan footer (nomor halaman)
    # (urutan skema OOXML: bordersDoNotSurroundHeader sebelum ...Footer, dan
    # keduanya sesudah w:zoom)
    settings = doc.settings.element
    sebelumnya = settings.find(qn("w:zoom"))
    for tag in ("w:bordersDoNotSurroundHeader", "w:bordersDoNotSurroundFooter"):
        el = settings.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            if sebelumnya is not None:
                sebelumnya.addnext(el)
            else:
                settings.insert(0, el)
        sebelumnya = el

    _page_setup(doc.sections[0])
    return doc


def _page_setup(section):
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(4.0)
    section.left_margin = Cm(4.0)
    section.bottom_margin = Cm(3.0)
    section.right_margin = Cm(3.0)
    section.header_distance = Cm(1.25)
    section.footer_distance = Cm(1.25)


def _clear(part):
    for p in part.paragraphs:
        for r in list(p.runs):
            r._element.getparent().remove(r._element)
    return part.paragraphs[0]


def _page_field_paragraph(part, align):
    p = _clear(part)
    p.alignment = align
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.first_line_indent = Cm(0)
    _field(_font(p.add_run()), " PAGE ")


_SESUDAH_PGNUM = ("cols", "formProt", "vAlign", "noEndnote", "titlePg", "textDirection",
                  "bidi", "rtlGutter", "docGrid", "printerSettings")


def _sisip_sectpr(sectPr, el, sesudahnya):
    """Sisipkan el tepat sebelum anak sectPr pertama yang menurut skema OOXML
    harus berada setelahnya."""
    for child in sectPr:
        if child.tag.split("}")[-1] in sesudahnya:
            child.addprevious(el)
            return
    sectPr.append(el)


def _pg_num(section, fmt, start=None):
    sectPr = section._sectPr
    pg = sectPr.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType")
        _sisip_sectpr(sectPr, pg, _SESUDAH_PGNUM)
    pg.set(qn("w:fmt"), fmt)
    if start is not None:
        pg.set(qn("w:start"), str(start))
    elif pg.get(qn("w:start")) is not None:
        del pg.attrib[qn("w:start")]


def _unlink(section):
    for part in (section.header, section.footer, section.first_page_header,
                 section.first_page_footer):
        part.is_linked_to_previous = False


def number_front(section, start=None, show=True):
    """Bagian awal: nomor romawi di tengah bawah (atau disembunyikan)."""
    _unlink(section)
    section.different_first_page_header_footer = False
    _clear(section.header)
    if show:
        _page_field_paragraph(section.footer, WD_ALIGN_PARAGRAPH.CENTER)
    else:
        _clear(section.footer)
    _pg_num(section, "lowerRoman", start)


def number_chapter(section, start=None):
    """Halaman bab: tengah bawah di halaman pertama, kanan atas di halaman lain."""
    _unlink(section)
    section.different_first_page_header_footer = True
    _clear(section.first_page_header)
    _page_field_paragraph(section.first_page_footer, WD_ALIGN_PARAGRAPH.CENTER)
    _page_field_paragraph(section.header, WD_ALIGN_PARAGRAPH.RIGHT)
    _clear(section.footer)
    _pg_num(section, "decimal", start)


def new_section(doc):
    """Section baru mewarisi pengaturan section sebelumnya, termasuk bingkai
    halaman, sehingga bingkai lembar pengesahan dibuang di sini."""
    s = doc.add_section(WD_SECTION.NEW_PAGE)
    _page_setup(s)
    for b in s._sectPr.findall(qn("w:pgBorders")):
        s._sectPr.remove(b)
    return s


def page_border(section):
    """Bingkai garis ganda (tebal di luar, tipis di dalam) di sekeliling teks,
    seperti bingkai lembar pengesahan pada template. Jaraknya ke teks mengikuti
    posisi bingkai template: sekitar 1 cm di atas, 0,6 cm di kiri, 0,8 cm di
    kanan, dan 0,2 cm di bawah teks, sehingga nomor halaman berada di luar bingkai."""
    sectPr = section._sectPr
    b = OxmlElement("w:pgBorders")
    b.set(qn("w:offsetFrom"), "text")
    for edge, space in (("top", "29"), ("left", "17"), ("bottom", "6"), ("right", "23")):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "thickThinSmallGap")
        el.set(qn("w:sz"), "24")
        el.set(qn("w:space"), space)
        el.set(qn("w:color"), "000000")
        b.append(el)
    _sisip_sectpr(sectPr, b, ("lnNumType", "pgNumType") + _SESUDAH_PGNUM)


# ------------------------------------------------------------ teks sebaris
def _tokens(text):
    """Pecah teks menjadi potongan (isi, tebal, miring, kode); kode bernilai
    "math" untuk rumus sebaris di antara tanda dolar.

    Ditulis sebagai pemindai sederhana, bukan satu regex, supaya teks tebal
    yang memuat kata miring seperti **ResNet50 *fold* 4** terbaca benar.
    """
    out, buf = [], []
    bold = italic = False
    i, n = 0, len(text)

    def flush():
        if buf:
            out.append(("".join(buf), bold, italic, False))
            buf.clear()

    while i < n:
        c = text[i]
        if c == "`":
            j = text.find("`", i + 1)
            if j > i:
                flush()
                out.append((text[i + 1:j], bold, italic, True))
                i = j + 1
                continue
        if c == "$" and i + 1 < n and text[i + 1] != " ":
            j = text.find("$", i + 1)
            if j > i + 1:
                flush()
                out.append((text[i + 1:j], bold, True, "math"))
                i = j + 1
                continue
        if text.startswith("**", i):
            flush(); bold = not bold; i += 2
            continue
        if c == "*" and (i + 1 < n and text[i + 1] != " " or italic):
            flush(); italic = not italic; i += 1
            continue
        buf.append(c)
        i += 1
    flush()
    return out


def _run_kode(paragraph, teks, size):
    """Run kode program. Tanda hubungnya dibuat tak-terpotong (w:noBreakHyphen),
    supaya Word tidak memenggal "pseudo-knn" atau "1e-3" di ujung baris."""
    bagian = teks.split("-")
    r = paragraph.add_run(bagian[0])
    for b in bagian[1:]:
        r._r.append(OxmlElement("w:noBreakHyphen"))
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = b
        r._r.append(t)
    _font(r, size=size, name=MONO_FONT)
    return r


def add_runs(paragraph, text, size=BODY_SIZE, force_bold=None, force_italic=None,
             underline=None):
    text = text.replace(" -- ", " – ")
    for isi, b, it, kode in _tokens(text):
        if not isi:
            continue
        if kode == "math":
            # rumus sebaris ditulis sebagai persamaan Word, sama dengan rumus bernomor
            paragraph._p.append(latex_ke_omml(isi))
            continue
        if kode:
            _run_kode(paragraph, isi, Pt(size.pt - 1.5))
        else:
            r = paragraph.add_run(isi)
            _font(r, size=size, bold=b if force_bold is None else (force_bold or b),
                  italic=it if force_italic is None else force_italic, underline=underline)
    return paragraph


# ----------------------------------------------------------------- paragraf
def body_paragraph(doc, text, indent=True, spacing=2.0, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                   italic=None):
    p = doc.add_paragraph()
    _pf(p, line=spacing, indent=INDENT if indent else Cm(0), align=align)
    add_runs(p, text, force_italic=italic)
    STATE["bawah_judul"] = False
    return p


def caption(doc, text, style):
    p = doc.add_paragraph(style=style)
    _pf(p, line=2.0, align=WD_ALIGN_PARAGRAPH.CENTER, keep=(style == "Caption Tabel"))
    add_runs(p, text)
    STATE["bawah_judul"] = False
    return p


def chapter_heading(doc, bab, title):
    """Satu paragraf Heading 1 berisi dua baris ("BAB I" lalu judulnya). Jarak
    4 spasi ke isi dihitung dari baris judul: satu baris 2 spasi ditambah 2 baris."""
    p = doc.add_paragraph(style="Heading 1")
    _pf(p, line=2.0, after=Pt(2 * BARIS), align=WD_ALIGN_PARAGRAPH.CENTER, keep=True)
    _font(p.add_run(bab.upper()), bold=True).add_break()
    add_runs(p, title.upper(), force_bold=True)
    STATE["bab"] = ROMAWI[bab.split()[1]]
    STATE["eq"] = 0
    STATE["bawah_judul"] = True
    return p


def front_heading(doc, text, in_toc=True):
    """Judul bagian awal, daftar pustaka, dan lampiran: 4 spasi ke isinya."""
    p = doc.add_paragraph(style="Heading 1" if in_toc else "Judul Tanpa Daftar")
    _pf(p, line=1.0, after=Pt(3 * BARIS), align=WD_ALIGN_PARAGRAPH.CENTER, keep=True)
    add_runs(p, text.upper() if not text.startswith("*") else text, force_bold=True)
    STATE["bawah_judul"] = True
    return p


def section_heading(doc, text, level=2):
    p = doc.add_paragraph(style=f"Heading {min(level, 3)}")
    gantung = Cm(1.0 if level == 2 else 1.25)
    pf = _pf(p, line=2.0, before=Pt(0) if STATE["bawah_judul"] else JARAK_SUBBAB,
             align=WD_ALIGN_PARAGRAPH.LEFT, keep=True)
    pf.left_indent = gantung
    pf.first_line_indent = -gantung
    pf.tab_stops.add_tab_stop(gantung)
    m = re.match(r"^((?:\d+\.)+\d*)\s+(.*)$", text)
    if m:
        num, rest = m.groups()
        _font(p.add_run(f"{num}\t"), bold=True)
        add_runs(p, rest, force_bold=True)
    else:
        add_runs(p, text, force_bold=True)
    STATE["bawah_judul"] = True
    return p


def judul_butir(doc, text, jarak=True):
    """Butir berjudul di bawah subbab, misalnya "A. Dataset Kaggle": tebal,
    rata kiri, nomor huruf/angka di 0 cm dan judulnya di 0,5 cm. Label bab pada
    sistematika penulisan (jarak=False) ditulis rapat seperti pada template."""
    p = doc.add_paragraph()
    pf = _pf(p, line=2.0, before=JARAK_SUBBAB if jarak and not STATE["bawah_judul"] else Pt(0),
             align=WD_ALIGN_PARAGRAPH.LEFT, keep=True)
    m = re.match(r"^([A-Z0-9]{1,2})\.\s+(.*)$", text)
    if m:
        pf.left_indent = Cm(0.5); pf.first_line_indent = Cm(-0.5)
        pf.tab_stops.add_tab_stop(Cm(0.5))
        _font(p.add_run(f"{m.group(1)}.\t"), bold=True)
        add_runs(p, m.group(2), force_bold=True)
    else:
        add_runs(p, text, force_bold=True)
    STATE["bawah_judul"] = True
    return p


def add_list_item(doc, marker, text, level=0, step=0.75):
    p = doc.add_paragraph()
    left = step * (level + 1)
    pf = _pf(p, line=2.0, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    pf.left_indent = Cm(left)
    pf.first_line_indent = Cm(-step)
    pf.tab_stops.add_tab_stop(Cm(left))
    add_runs(p, f"{marker}\t{text}")
    STATE["bawah_judul"] = False
    return p


def _set_cell_border(cell, sz="4"):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), sz); el.set(qn("w:color"), "000000")
        borders.append(el)
    tcPr.append(borders)


def _gap(doc):
    """Satu baris kosong berspasi tunggal sesudah tabel atau kotak kode."""
    g = doc.add_paragraph()
    _pf(g, line=1.0)
    return g


def add_code_block(doc, lines):
    """Kode program di dalam kotak satu sel seperti pada template: 9 pt, 1 spasi.

    Baris yang lebih panjang dari lebar kotak dilipat Word. Lipatannya dibuat
    menjorok empat karakter lebih dalam dari indentasi baris asalnya (paling
    jauh sepertiga lebar kotak), supaya tidak terbaca sebagai baris baru di
    kolom paling kiri, karena indentasi pada Python bermakna."""
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    lebar_huruf = MONO_EM * CODE_SIZE.pt
    muat = int((Cm(TEXT_WIDTH_CM - 0.381).pt) / lebar_huruf) - 1   # karakter per baris
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    cell = t.cell(0, 0)
    cell.width = Cm(TEXT_WIDTH_CM)
    _set_cell_border(cell)
    for k, line in enumerate(lines):
        p = cell.paragraphs[0] if k == 0 else cell.add_paragraph()
        teks = line.rstrip()
        isi = teks.lstrip(" ")
        n = len(teks) - len(isi)
        if isi and n + len(isi.split()[0]) > muat:
            # baris lanjutan rata-sejajar di dalam kurung yang tidak muat: geser ke
            # kiri seperlunya (spasi di dalam kurung tidak bermakna bagi Python)
            n = max(0, muat - len(isi))
            teks = " " * n + isi
        pf = _pf(p, line=1.0, align=WD_ALIGN_PARAGRAPH.LEFT)
        lipatan = Pt(min(n + 4, muat // 3) * lebar_huruf)
        pf.left_indent, pf.first_line_indent = lipatan, -lipatan
        _run_kode(p, teks, CODE_SIZE)
    _gap(doc)
    STATE["bawah_judul"] = False


# ------------------------------------------------------------- gambar/rumus
_XSL = {}


def latex_ke_omml(latex: str):
    """LaTeX -> MathML (latex2mathml) -> OMML (MML2OMML.XSL bawaan Office), agar
    persamaan tampil sebagai persamaan Word asli seperti pada template."""
    from latex2mathml.converter import convert
    from lxml import etree
    if "x" not in _XSL:
        _XSL["x"] = etree.XSLT(etree.parse(str(MML2OMML)))
    omml = _XSL["x"](etree.fromstring(convert(latex.strip())))
    root = omml.getroot()
    if root.tag.endswith("oMathPara"):
        root = root.find(qn("m:oMath"))
    from docx.oxml import parse_xml
    return parse_xml(etree.tostring(root))


def add_equation(doc, latex):
    """Persamaan di tengah dan nomornya rata kanan, misalnya (2.1)."""
    STATE["eq"] += 1
    nomor = f"({STATE['bab']}.{STATE['eq']})"
    p = doc.add_paragraph()
    pf = _pf(p, line=1.5, before=6, after=6, align=WD_ALIGN_PARAGRAPH.LEFT)
    pf.tab_stops.add_tab_stop(Cm(TEXT_WIDTH_CM / 2), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(Cm(TEXT_WIDTH_CM), WD_TAB_ALIGNMENT.RIGHT)
    _font(p.add_run("\t"))
    p._p.append(latex_ke_omml(latex))
    _font(p.add_run(f"\t{nomor}"))
    STATE["bawah_judul"] = False
    return nomor


def add_figure(doc, md_path, cap_text):
    fname, width_cm = FIGURES[md_path]
    p = doc.add_paragraph()
    _pf(p, line=1.0, before=6, align=WD_ALIGN_PARAGRAPH.CENTER, keep=True)
    p.add_run().add_picture(str(FIG / fname), width=Cm(min(width_cm, TEXT_WIDTH_CM)))
    caption(doc, cap_text, "Caption Gambar")


# ------------------------------------------------------------------- tabel
CELL_IMG_RE = re.compile(r"^!\[([\d.,]*)\]\((.+?)\)$")


def _shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def _row_props(row, header=False):
    trPr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit"); trPr.append(cant)
    if header:
        th = OxmlElement("w:tblHeader"); trPr.append(th)


def _plain(cell):
    return re.sub(r"[*`]", "", cell).strip()


_FONT_FILES = {(False, False): "times.ttf", (True, False): "timesbd.ttf",
               (False, True): "timesi.ttf", (True, True): "timesbi.ttf"}
_FONT_CACHE: dict = {}


def _lebar_cm(teks, size_pt, bold=False, italic=False):
    """Lebar teks sebenarnya dalam cm, diukur dari berkas huruf Times New Roman."""
    from PIL import ImageFont
    key = (bold, italic)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(f"C:/Windows/Fonts/{_FONT_FILES[key]}", 100)
    return _FONT_CACHE[key].getlength(teks) / 100 * size_pt * 0.03528


def _lebar_sel(teks, size_pt, header=False, utuh=False):
    """Lebar minimum sel: kata terpanjang (bila boleh dibungkus) atau seluruh
    isi (bila sel pendek yang tidak boleh terpotong, misalnya angka).

    Satu kata dapat terdiri atas beberapa potongan bergaya berbeda, misalnya
    "(" tegak lalu "Feature" miring; lebarnya dijumlahkan karena Word hanya
    memenggal baris pada spasi atau sesudah tanda hubung."""
    def ukur(s, b, it, kode):
        if kode and kode != "math":   # kode sebaris: Consolas, 1,5 pt lebih kecil
            return len(s) * MONO_EM * (size_pt - 1.5) * 0.03528
        return _lebar_cm(s, size_pt, b, it)

    total = kata_max = berjalan = 0.0
    for isi, b, it, kode in _tokens(teks):
        b = b or header
        for w in re.split(r"([^\S ]+)", isi):   # spasi tak-terpotong menyambung kata
            if not w:
                continue
            total += ukur(w, b, it, kode)
            if w.isspace() and " " not in w:
                berjalan = 0.0
                continue
            # Word boleh memenggal baris sesudah tanda hubung (kecuali pada kode),
            # sehingga k-nearest-neighbor cukup selebar potongan terpanjangnya
            bagian = [w] if kode else re.split(r"(?<=-)", w)
            for k, potongan in enumerate(bagian):
                if k:
                    berjalan = 0.0
                if potongan:
                    berjalan += ukur(potongan, b, it, kode)
                    kata_max = max(kata_max, berjalan)
    return total if utuh else kata_max


PAD_CM = 0.45        # margin dalam sel baku (2 x 0,19 cm) ditambah ruang aman
PAD_RAPAT_CM = 0.25  # margin dalam sel dirapatkan (2 x 0,08 cm) untuk tabel lebar


def _is_numeric_col(rows, j):
    body = [_plain(r[j]) for r in rows[1:] if j < len(r)]
    return bool(body) and all(re.fullmatch(r"[\d.,%\-+−×\s/()e]*", c) and len(c) <= 18 for c in body)


def _is_center_col(rows, j):
    body = [_plain(r[j]) for r in rows[1:] if j < len(r)]
    if not body:
        return False
    numeric = all(re.fullmatch(r"[\d.,%\-+−×\s/()e]*", c) and len(c) <= 18 for c in body)
    short = all(len(c) <= 16 for c in body)
    return numeric or short


def _column_widths(rows, size_pt, pad=PAD_CM):
    """Kolom angka dan kolom berisi teks pendek diberi lebar secukupnya agar
    isinya tidak terpotong; sisa lebar halaman dibagikan kepada kolom berisi
    teks panjang sebanding panjang teksnya."""
    PAD_CM = pad
    ncol = len(rows[0])
    need, bobot = [], []
    for j in range(ncol):
        col = [r[j] if j < len(r) else "" for r in rows]
        imgs = [CELL_IMG_RE.match(c.strip()) for c in col[1:]]
        if any(imgs):
            w = max(float(m.group(1).replace(",", ".") or 3.2) for m in imgs if m)
            need.append(max(w + PAD_CM, _lebar_sel(col[0], size_pt, header=True) + PAD_CM))
            bobot.append(0.0)
            continue
        pendek = _is_center_col(rows, j)
        angka = _is_numeric_col(rows, j)
        n_hdr = _lebar_sel(col[0], size_pt, header=True)
        n_body = max((_lebar_sel(c, size_pt, utuh=angka) for c in col[1:]), default=0)
        need.append(max(n_hdr, n_body) + PAD_CM)
        bobot.append(0.0 if pendek else
                     sum(_lebar_sel(c, size_pt, utuh=True) for c in col[1:]) / max(len(col) - 1, 1))
    lebar = list(need)
    sisa = TEXT_WIDTH_CM - sum(lebar)
    if sisa > 0:
        tot = sum(bobot)
        if tot > 0:
            lebar = [w + sisa * b / tot for w, b in zip(lebar, bobot)]
        else:
            lebar = [w + sisa / ncol for w in lebar]
    return lebar, sum(need) <= TEXT_WIDTH_CM + 1e-6


def _perkiraan_tinggi(rows, widths, size_pt):
    """Perkiraan tinggi tabel dalam cm: jumlah baris teks terbanyak pada setiap
    baris tabel dikali tinggi satu baris spasi tunggal ditambah margin sel."""
    baris_cm = size_pt * 1.15 * 0.03528
    total = 0.0
    for r in rows:
        terbanyak = 1
        for j, teks in enumerate(r[:len(widths)]):
            isi = _lebar_sel(teks, size_pt, utuh=True)
            terbanyak = max(terbanyak, int(isi // max(widths[j] - 0.4, 0.5)) + 1)
        total += terbanyak * baris_cm + 0.07
    return total


def add_table(doc, rows):
    # angka dalam kurung tidak dipisahkan dari katanya, misalnya "Kaggle (4)",
    # supaya "(4)" tidak jatuh sendirian ke baris berikutnya di dalam sel
    rows = [rows[0]] + [[re.sub(r"(?<=\w) (?=\(\d+\))", " ", c) for c in r] for r in rows[1:]]
    header, *body = rows
    ncol = len(header)
    # isi tabel memakai huruf standar 12 pt; tabel yang terlalu lebar lebih dulu
    # dirapatkan margin dalam selnya, baru kemudian hurufnya diturunkan satu
    # tingkat, agar tidak ada kata yang terpotong di tengah
    for size in TABLE_SIZES:
        for pad in (PAD_CM, PAD_RAPAT_CM):
            widths, muat = _column_widths(rows, size.pt, pad)
            if muat:
                break
        if muat:
            break
    if size.pt < 12 or pad != PAD_CM:
        print(f"  [tabel {size.pt:g} pt{', sel rapat' if pad != PAD_CM else ''}] "
              f"{' | '.join(_plain(h)[:14] for h in header)}", flush=True)
    if not muat:
        skala = TEXT_WIDTH_CM / sum(widths)
        widths = [w * skala for w in widths]
        print(f"  [PERINGATAN: tabel masih terlalu lebar, diskala {skala:.2f}]", flush=True)
    center = [_is_center_col(rows, j) for j in range(ncol)]
    # tabel yang muat dalam satu halaman dijaga tetap utuh agar tidak ada baris
    # yang tertinggal sendirian di bawah halaman sebelumnya
    utuh = (not any(CELL_IMG_RE.match(c.strip()) for r in rows[1:] for c in r)
            and _perkiraan_tinggi(rows, widths, size.pt) <= 18.0)
    table = doc.add_table(rows=len(rows), cols=ncol)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tblPr = table._tbl.tblPr
    layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed"); tblPr.append(layout)
    if pad != PAD_CM:
        mar = OxmlElement("w:tblCellMar")
        for edge in ("left", "right"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:w"), str(int(0.08 / 2.54 * 1440))); el.set(qn("w:type"), "dxa")
            mar.append(el)
        tblPr.append(mar)
    for j, w in enumerate(widths):
        for cell in table.columns[j].cells:
            cell.width = Cm(w)

    for i, row in enumerate(rows):
        _row_props(table.rows[i], header=(i == 0))
        for j in range(ncol):
            text = row[j] if j < len(row) else ""
            cell = table.cell(i, j)
            cell.text = ""
            cell.vertical_alignment = (WD_CELL_VERTICAL_ALIGNMENT.CENTER if i == 0
                                       else WD_CELL_VERTICAL_ALIGNMENT.TOP)
            p = cell.paragraphs[0]
            _pf(p, line=1.0, before=1, after=1)
            m = CELL_IMG_RE.match(text.strip())
            if m and i > 0:
                w_cm = float(m.group(1).replace(",", ".") or 3.2)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(str(SRC / m.group(2)), width=Cm(w_cm))
            elif i == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_runs(p, text, size=size, force_bold=True)
            else:
                # rata kiri-kanan hanya untuk paragraf panjang seperti kolom
                # analisis; sel pendek rata kiri agar tidak muncul celah lebar
                paragraf = len(_plain(text)) > 150
                p.alignment = (WD_ALIGN_PARAGRAPH.CENTER if center[j]
                               else WD_ALIGN_PARAGRAPH.JUSTIFY if paragraf
                               else WD_ALIGN_PARAGRAPH.LEFT)
                add_runs(p, text, size=size)
            _set_cell_border(cell)
            if i == 0:
                _shade(cell, HEADER_FILL)
            # baris kepala selalu ikut pindah bersama baris isi pertamanya, supaya
            # judul dan kepala tabel tidak tertinggal sendirian di dasar halaman
            if (utuh and i < len(rows) - 1) or i == 0:
                for par in cell.paragraphs:
                    par.paragraph_format.keep_with_next = True
    _gap(doc)
    STATE["bawah_judul"] = False
    return table


# ---------------------------------------------------------- parser markdown
CAPTION_TAB_RE = re.compile(r"^(Tabel\s+[\d.]+\s+.*)$")
IMG_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)$")
OL_RE = re.compile(r"^(\d+)\.\s+(.*)$")
AL_RE = re.compile(r"^([a-z])\.\s+(.*)$")
UL_RE = re.compile(r"^[-*]\s+(.*)$")
BOLD_LINE_RE = re.compile(r"^\*\*([^*].*?)\*\*$")
LABEL_KODE_RE = re.compile(r"^\*Script\*/kode program ")


def _hanya_tebal(s):
    """Baris yang seluruhnya tebal, misalnya **A. Dataset Kaggle** atau
    **BAB I PENDAHULUAN**; teks tebal di tengah kalimat tidak termasuk."""
    m = BOLD_LINE_RE.match(s)
    return m.group(1) if m and "**" not in m.group(1).replace("***", "") else None


def parse_markdown(doc, path: Path, on_chapter, pustaka=False, lampiran=False, list_step=0.75):
    """on_chapter() dipanggil sebelum judul bab ditulis, agar setiap bab dibuka
    pada section baru dengan penomoran halamannya sendiri."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    pending_bab = None
    pending_table_caption = None
    pending_figure = None
    label_kode = False

    while i < len(lines):
        line = lines[i].rstrip()
        s = line.strip()
        if not s:
            i += 1
            continue

        if line.startswith("# "):
            text = line[2:].strip()
            if re.match(r"^BAB\s+[IVX]+$", text):
                pending_bab = text
                i += 1
                continue
            on_chapter()
            if pending_bab:
                chapter_heading(doc, pending_bab, text)
                pending_bab = None
            else:
                front_heading(doc, text)
            i += 1
            continue

        if line.startswith("### "):
            section_heading(doc, line[4:].strip(), level=3); i += 1; continue
        if line.startswith("## "):
            section_heading(doc, line[3:].strip(), level=2); i += 1; continue

        m = IMG_RE.match(s)
        if m:
            if m.group(2) not in FIGURES:
                raise KeyError(f"{path.name}: gambar belum terdaftar di FIGURES -> {m.group(2)}")
            pending_figure = m.group(2); i += 1; continue

        if pending_figure:
            cap = s.strip("*")
            if not cap.startswith("Gambar"):
                raise ValueError(f"{path.name}:{i + 1}: gambar tanpa keterangan -> {pending_figure}")
            add_figure(doc, pending_figure, cap)
            pending_figure = None
            i += 1
            continue

        if CAPTION_TAB_RE.match(s):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].lstrip().startswith("|"):
                pending_table_caption = s
                i += 1
                continue

        if line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if not pending_table_caption:
                raise ValueError(f"{path.name}:{i}: tabel tanpa judul tabel di atasnya")
            caption(doc, pending_table_caption, "Caption Tabel")
            pending_table_caption = None
            add_table(doc, rows)
            continue

        if s.startswith("$$"):
            if s.endswith("$$") and len(s) > 4:
                latex = s[2:-2]; i += 1
            else:
                parts = [s[2:]]; i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    parts.append(lines[i]); i += 1
                if i < len(lines):
                    parts.append(lines[i].strip()[:-2]); i += 1
                latex = " ".join(parts)
            add_equation(doc, latex)
            continue

        if s.startswith("```"):
            if not label_kode:
                raise ValueError(f"{path.name}:{i + 1}: kode program tanpa keterangan nama kodenya")
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            i += 1
            add_code_block(doc, code)
            label_kode = False
            continue
        label_kode = False

        if set(s) <= {"-"} and len(s) >= 3:
            i += 1
            continue

        if lampiran and re.match(r"^Lampiran \d+ ", s):
            p = doc.add_paragraph(style="Judul Lampiran")
            _pf(p, line=2.0, align=WD_ALIGN_PARAGRAPH.LEFT, keep=True)
            add_runs(p, s, force_bold=True)
            i += 1
            continue

        tebal = _hanya_tebal(s)
        if tebal is not None:
            label_bab = tebal.startswith("BAB ")
            judul_butir(doc, tebal, jarak=not label_bab)
            # uraian di bawah label bab pada sistematika penulisan tidak berinden
            STATE["tanpa_inden"] = label_bab
            i += 1
            continue

        m = OL_RE.match(s)
        if m:
            add_list_item(doc, f"{m.group(1)}.", m.group(2), step=list_step); i += 1; continue
        m = AL_RE.match(s)
        if m:
            level = 1 if line.startswith((" ", "\t")) else 0
            add_list_item(doc, f"{m.group(1)}.", m.group(2), level=level, step=list_step)
            i += 1
            continue
        m = UL_RE.match(s)
        if m and not s.startswith("**"):
            add_list_item(doc, "\u2022", m.group(1), step=list_step); i += 1; continue

        # keterangan nama kode program ditulis rata kiri tanpa inden, seperti template
        if LABEL_KODE_RE.match(s):
            body_paragraph(doc, s, indent=False, align=WD_ALIGN_PARAGRAPH.LEFT)
            label_kode = True
            i += 1
            continue

        if pustaka:
            # daftar pustaka template: inden gantung 1 cm, spasi tunggal, jarak 8 pt.
            # Tautan DOI boleh dipotong sesudah "/" (spasi lebar-nol), supaya
            # baris rata kiri-kanan sebelum tautan tidak meregang renggang.
            s = re.sub(r"https?://\S+",
                       lambda m: re.sub(r"(?<!/)/(?!/)", "/​", m.group(0)), s)
            pr = body_paragraph(doc, s, indent=False, spacing=1.0)
            pr.paragraph_format.left_indent = Cm(1.0)
            pr.paragraph_format.first_line_indent = Cm(-1.0)
            pr.paragraph_format.space_after = Pt(8)
            pr.paragraph_format.keep_together = True   # satu entri tidak terbelah dua halaman
        elif lampiran or s.startswith("http"):
            body_paragraph(doc, s, indent=False, align=WD_ALIGN_PARAGRAPH.LEFT)
        else:
            body_paragraph(doc, s, indent=not STATE["tanpa_inden"])
            STATE["tanpa_inden"] = False
        i += 1


# ------------------------------------------------------ bagian awal (md)
def _blok(front, judul):
    """Isi satu bagian 00_Halaman_Depan_dan_Abstrak.md, dari judulnya hingga ---."""
    m = re.search(rf"^#+ {re.escape(judul)}\s*$(.*?)(?=^---\s*$|\Z)", front, re.M | re.S)
    if not m:
        raise KeyError(f"bagian '{judul}' tidak ada di halaman depan")
    return m.group(1).strip()


def _kelompok(blok):
    """Pecah blok menjadi kelompok baris yang dipisahkan baris kosong."""
    return [[b.strip() for b in g.splitlines() if b.strip()]
            for g in re.split(r"\n\s*\n", blok) if g.strip()]


def _centred(doc, text="", bold=False, italic=None, line=1.5, after=0, before=0, size=BODY_SIZE):
    p = doc.add_paragraph()
    _pf(p, line=line, before=before, after=after, align=WD_ALIGN_PARAGRAPH.CENTER)
    if text:
        add_runs(p, text, size=size, force_bold=bold or None, force_italic=italic)
    return p


def _baris_ganda(doc, lines, line=1.5, bold=False):
    """Beberapa baris dalam satu paragraf yang dipisah pindah baris."""
    p = _centred(doc, line=line)
    for k, t in enumerate(lines):
        add_runs(p, t, force_bold=bold or None)
        if k < len(lines) - 1:
            p.runs[-1].add_break()
    return p


def title_page(doc, front):
    judul, skripsi, diajukan, identitas, institusi = _kelompok(_blok(front, "HALAMAN JUDUL"))
    _centred(doc, judul[0])
    _centred(doc); _centred(doc)
    _centred(doc, skripsi[0])
    _centred(doc); _centred(doc)
    _baris_ganda(doc, diajukan)
    _centred(doc); _centred(doc)
    _baris_ganda(doc, identitas)
    _centred(doc)
    p = _centred(doc, line=1.0, before=10)
    p.add_run().add_picture(str(LOGO), width=Cm(4.42))
    _centred(doc); _centred(doc)
    _baris_ganda(doc, institusi, line=1.0)


def _penguji(doc, nomor, nama, peran, nip, first):
    """Satu anggota tim penguji: nama bergaris bawah dan perannya, lalu NIP dan
    garis titik tempat tanda tangan di sisi kanan, seperti pada template."""
    p = doc.add_paragraph()
    pf = _pf(p, line=1.0, before=0 if first else 6, align=WD_ALIGN_PARAGRAPH.LEFT)
    pf.left_indent = Cm(0.63); pf.first_line_indent = Cm(-0.63)
    pf.tab_stops.add_tab_stop(Cm(0.63))
    pf.tab_stops.add_tab_stop(Cm(7.6))
    _font(p.add_run(f"{nomor}.\t"))
    _font(p.add_run(nama), underline=True)
    _font(p.add_run(f"\t{peran}"))
    q = doc.add_paragraph()
    qf = _pf(q, line=1.0, align=WD_ALIGN_PARAGRAPH.LEFT)
    qf.left_indent = Cm(0.63)
    qf.tab_stops.add_tab_stop(Cm(TEXT_WIDTH_CM), WD_TAB_ALIGNMENT.RIGHT)
    _font(q.add_run(f"NIP. {nip}\t………………………"))


def lembar_pengesahan(doc, front):
    blok = _blok(front, "LEMBAR PENGESAHAN")
    tabel = [b for b in blok.splitlines() if b.strip().startswith("|")]
    teks = "\n".join(b for b in blok.splitlines() if not b.strip().startswith("|"))
    g = _kelompok(teks)
    skripsi, judul_id, judul_en, disusun, identitas, sidang, susunan = g
    _centred(doc, skripsi[0], line=1.0)
    _centred(doc, line=1.0)
    _centred(doc, judul_id[0])
    _centred(doc, line=1.0)
    _centred(doc, judul_en[0])
    _centred(doc)
    _centred(doc, disusun[0])
    _centred(doc)
    _baris_ganda(doc, identitas)
    _centred(doc)
    _baris_ganda(doc, sidang)
    _centred(doc, line=1.0)
    _centred(doc, susunan[0])
    _centred(doc, line=1.0)
    anggota = [[c.strip() for c in b.strip().strip("|").split("|")] for b in tabel[2:]]
    for k, (no, nama, peran, nip) in enumerate(anggota):
        _penguji(doc, no, nama, peran, nip, first=(k == 0))


def kata_pengantar(doc, front):
    front_heading(doc, "Kata Pengantar")
    blok = _blok(front, "KATA PENGANTAR")
    paragraf = [b.strip() for b in blok.split("\n\n") if b.strip()]
    for para in paragraf:
        baris = [b.strip() for b in para.splitlines() if b.strip()]
        if all(OL_RE.match(b) for b in baris):
            for b in baris:
                m = OL_RE.match(b)
                add_list_item(doc, f"{m.group(1)}.", m.group(2), step=1.0)
        elif para.startswith("Jatinangor,") or para == "Penulis":
            # tanggal dan penulis di sisi kanan, satu baris kosong di antaranya
            p = body_paragraph(doc, para, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
            p.paragraph_format.left_indent = Cm(8.5)
            if para == "Penulis":
                p.paragraph_format.space_before = Pt(2 * BARIS)
        else:
            body_paragraph(doc, para)


def abstrak(doc, front, key, head, english=False):
    front_heading(doc, head)
    blok = _blok(front, key)
    for para in [x.strip() for x in blok.split("\n\n") if x.strip()]:
        is_kw = para.startswith("**Kata Kunci**") or para.startswith("**Keywords**")
        if is_kw:
            # label kata kunci tebal; isi abstract berbahasa Inggris seluruhnya miring
            p = body_paragraph(doc, para, indent=False, spacing=1.5,
                               italic=True if english else None)
            p.paragraph_format.space_before = Pt(BARIS)
        else:
            body_paragraph(doc, para, spacing=1.0, italic=True if english else None)


def toc_list(doc, head, instr):
    front_heading(doc, head)
    p = doc.add_paragraph()
    _pf(p, line=1.5)
    _field(_font(p.add_run()), instr)


# ----------------------------------------------------------------- rakitan
def _kutipan(teks):
    """(nama keluarga, tahun) dari setiap kutipan di teks."""
    nama = r"[A-Z][A-Za-z\-]+"
    return {(m.group(1), m.group(3)) for m in
            re.finditer(rf"({nama})(?: et al\.| & ({nama}))?,? \(?((?:19|20)\d\d)\)?", teks)}


def pustaka_bab(bab_files, out: Path):
    """Daftar pustaka yang hanya memuat sumber yang dikutip pada bab terpilih."""
    dikutip = set()
    for f in bab_files:
        dikutip |= _kutipan((SRC / f).read_text(encoding="utf-8"))
    baris = (SRC / "DAFTAR_PUSTAKA.md").read_text(encoding="utf-8").splitlines()
    simpan = [baris[0], ""]
    for b in baris[1:]:
        m = re.match(r"^([^,(]+),.*?\(((?:19|20)\d\d)\)", b.strip())
        if m and (m.group(1).strip(), m.group(2)) in dikutip:
            simpan += [b.strip(), ""]
    out.write_text("\n".join(simpan), encoding="utf-8")
    return out


def build(scope="lengkap"):
    doc = setup_document()
    STATE.update(bab=0, eq=0, bawah_judul=False, tanpa_inden=False)
    front = (SRC / "00_Halaman_Depan_dan_Abstrak.md").read_text(encoding="utf-8")
    bab_files = BAB if scope == "lengkap" else BAB[:1]

    # halaman judul: dihitung sebagai halaman i tetapi nomornya tidak tampil
    title_page(doc, front)
    number_front(doc.sections[0], start=1, show=False)

    # lembar pengesahan berbingkai, halaman ii
    s = new_section(doc)
    number_front(s)
    page_border(s)
    lembar_pengesahan(doc, front)

    s = new_section(doc)
    number_front(s)
    kata_pengantar(doc, front)
    doc.add_page_break(); abstrak(doc, front, "ABSTRAK", "Abstrak")
    doc.add_page_break(); abstrak(doc, front, "ABSTRACT", "*ABSTRACT*", english=True)
    doc.add_page_break(); toc_list(doc, "Daftar Isi", r' TOC \o "1-3" \h \z \u ')
    teks_bab = "\n".join((SRC / f).read_text(encoding="utf-8") for f in bab_files)
    if re.search(r"^Tabel \d+\.\d+ ", teks_bab, re.M):
        doc.add_page_break(); toc_list(doc, "Daftar Tabel", r' TOC \h \z \t "Caption Tabel,9" ')
    if re.search(r"^!\[", teks_bab, re.M):
        doc.add_page_break(); toc_list(doc, "Daftar Gambar", r' TOC \h \z \t "Caption Gambar,9" ')
    if scope == "lengkap":
        doc.add_page_break(); toc_list(doc, "Daftar Lampiran", r' TOC \h \z \t "Judul Lampiran,9" ')

    state = {"first": True}

    def on_chapter():
        s = new_section(doc)
        number_chapter(s, start=1 if state["first"] else None)
        state["first"] = False

    for name in bab_files:
        parse_markdown(doc, SRC / name, on_chapter,
                       list_step=1.0 if name.startswith("BAB_V") else 0.75)
    if scope == "lengkap":
        parse_markdown(doc, SRC / "DAFTAR_PUSTAKA.md", on_chapter, pustaka=True)
        parse_markdown(doc, SRC / "LAMPIRAN.md", on_chapter, lampiran=True)
        # riwayat hidup dikosongkan untuk diisi penulis, seperti pada template
        on_chapter()
        front_heading(doc, "Riwayat Hidup", in_toc=False)
        out = SRC / "Skripsi_Lengkap.docx"
    else:
        import tempfile
        tmp = Path(tempfile.gettempdir()) / "DAFTAR_PUSTAKA_BAB_I.md"
        parse_markdown(doc, pustaka_bab(bab_files, tmp), on_chapter, pustaka=True)
        out = SRC / "Skripsi_Bab_I.docx"

    try:
        doc.save(out)
        print(f"Saved: {out}", flush=True)
    except PermissionError:
        alt = out.with_name(out.stem + "_BARU.docx")
        doc.save(alt)
        print(f"Saved: {alt}  (berkas utama terkunci Word)", flush=True)


if __name__ == "__main__":
    build("bab1" if len(sys.argv) > 1 and sys.argv[1] == "bab1" else "lengkap")
