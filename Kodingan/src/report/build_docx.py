"""Build the formatted skripsi as a single .docx from the markdown chapters.

Formatting follows standard Indonesian skripsi conventions (and the example
thesis used as reference): A4, margins 4-3-3-3 cm, Times New Roman 12 pt,
double-spaced justified body with a first-line indent, figure captions
centred below the figure, table captions centred above the table, code
listings in a bordered monospace box, and page numbers in the footer.
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SRC = Path("D:/skripsi/Naskah Skripsi")
FIG = Path("D:/skripsi/Kodingan/outputs/figures/final")
OUT = SRC / "Skripsi_Lengkap.docx"

BODY_FONT = "Times New Roman"
MONO_FONT = "Consolas"
BODY_SIZE = Pt(12)
INDENT = Cm(1.27)

# markdown image path -> (final figure file, width in cm)
FIGURES = {
    "Gambar/Gambar_2.1_Arsitektur_CNN.png": ("gambar_2_1_arsitektur_cnn.png", 15.0),
    "Gambar/Gambar_2.2_Blok_MBConv_EfficientNet.png": ("gambar_2_2_blok_mbconv.png", 15.0),
    "Gambar/Gambar_2.3_Blok_Residual_ResNet50.png": ("gambar_2_3_blok_residual.png", 14.0),
    "Gambar/Gambar_3.1_Diagram_Alur_Penelitian.png": ("gambar_3_1_alur_penelitian.png", 9.5),
    "Gambar/Gambar_3.2_Sampel_Kaggle_Al-Yasriy.png": ("gambar_3_2_sampel_kaggle_alyasriy.png", 9.5),
    "Gambar/Gambar_3.3_Sampel_Kaggle_Rathi.png": ("gambar_3_3_sampel_kaggle_rathi.png", 9.5),
    "Gambar/Gambar_3.4_Sampel_LIDC.png": ("gambar_3_4_sampel_lidc.png", 9.5),
    "Gambar/Gambar_3.5_Use_Case_Diagram.png": ("gambar_3_5_use_case.png", 14.5),
    "Gambar/Gambar_4.1_Kurva_ResNet_Fold4.png": ("gambar_4_1_kurva_resnet_fold4.png", 15.0),
    "Gambar/Gambar_4.2_Kontribusi_Finetuning.png": ("gambar_4_2_kontribusi_finetuning.png", 14.5),
    "Gambar/Gambar_4.3_Kontribusi_Augmentasi.png": ("gambar_4_3_kontribusi_augmentasi.png", 13.0),
    "Gambar/Gambar_4.4_Kontribusi_Ensemble.png": ("gambar_4_4_kontribusi_ensemble.png", 15.0),
    "Gambar/Gambar_4.5_Confusion_Matrix_Stacking.png": ("gambar_4_5_confusion_matrix.png", 11.0),
    "Gambar/Gambar_4.6_Kurva_ROC_Stacking.png": ("gambar_4_6_kurva_roc.png", 11.5),
    "Gambar/Gambar_4.7_Sweep_Ambang_Keputusan.png": ("gambar_4_7_sweep_ambang.png", 14.0),
    "Gambar/Gambar_4.8_Performa_Per_Sumber.png": ("gambar_4_8_performa_per_sumber.png", 12.5),
    "Gambar/Gambar_4.9_Aplikasi_Beranda.png": ("gambar_4_9_aplikasi_beranda.png", 15.0),
    "Gambar/Gambar_4.10_Aplikasi_Prediksi.png": ("gambar_4_10_aplikasi_prediksi.png", 15.0),
    "Gambar/Gambar_4.11_Aplikasi_Tolak.png": ("gambar_4_11_aplikasi_tolak.png", 15.0),
    "Gambar/Gambar_4.12_Uji_Validasi_Input.png": ("gambar_4_12_uji_validasi_input.png", 15.0),
}


# --------------------------------------------------------------------- setup
def setup_document():
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = BODY_FONT
    style.font.size = BODY_SIZE
    style.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    pf = style.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 2.0

    # Heading styles drive the Daftar Isi field, so they must be the real
    # built-in styles rather than manually bolded paragraphs.
    for name, size, align, before, after in (
        ("Heading 1", 14, WD_ALIGN_PARAGRAPH.CENTER, 0, 24),
        ("Heading 2", 12, WD_ALIGN_PARAGRAPH.LEFT, 12, 0),
        ("Heading 3", 12, WD_ALIGN_PARAGRAPH.LEFT, 12, 0),
    ):
        st = doc.styles[name]
        st.font.name = BODY_FONT
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        st.paragraph_format.alignment = align
        st.paragraph_format.line_spacing = 2.0
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True

    # Separate caption styles so Daftar Gambar and Daftar Tabel can be built
    # from two independent TOC fields.
    for name in ("Caption Gambar", "Caption Tabel"):
        st = doc.styles.add_style(name, 1)  # 1 = WD_STYLE_TYPE.PARAGRAPH
        st.base_style = doc.styles["Normal"]
        st.font.name = BODY_FONT
        st.font.size = Pt(12)
        st.font.bold = False
        st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        st.paragraph_format.line_spacing = 1.0
        st.paragraph_format.first_line_indent = Cm(0)
        st.quick_style = False

    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.left_margin = Cm(4.0)
        section.right_margin = Cm(3.0)
        section.top_margin = Cm(3.0)
        section.bottom_margin = Cm(3.0)
    return doc


def add_page_number_footer(section, fmt="decimal", start=1):
    """Footer with a centred PAGE field."""
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    run = p.add_run()
    run.font.name = BODY_FONT
    run.font.size = BODY_SIZE
    fld_begin = OxmlElement("w:fldChar"); fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar"); fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin); run._r.append(instr); run._r.append(fld_end)

    sectPr = section._sectPr
    pgNumType = sectPr.find(qn("w:pgNumType"))
    if pgNumType is None:
        pgNumType = OxmlElement("w:pgNumType")
        sectPr.append(pgNumType)
    pgNumType.set(qn("w:fmt"), fmt)
    pgNumType.set(qn("w:start"), str(start))


# ------------------------------------------------------------ inline parsing
TOKEN_RE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\$[^$]+\$)")


def add_runs(paragraph, text):
    """Write text into a paragraph, honouring **bold**, *italic*, `code` and
    inline $math$ (rendered as an italic symbol, which is all the inline math
    in this manuscript needs)."""
    text = text.replace(" -- ", " – ")  # typographic en dash
    for part in TOKEN_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = paragraph.add_run(part[2:-2]); r.bold = True
        elif part.startswith("*") and part.endswith("*"):
            r = paragraph.add_run(part[1:-1]); r.italic = True
        elif part.startswith("`") and part.endswith("`"):
            r = paragraph.add_run(part[1:-1])
            r.font.name = MONO_FONT
            r.font.size = Pt(10.5)
        elif part.startswith("$") and part.endswith("$") and len(part) > 2:
            r = paragraph.add_run(part[1:-1])
            r.italic = True
        else:
            r = paragraph.add_run(part)
        r.font.name = r.font.name or BODY_FONT
    return paragraph


def body_paragraph(doc, text, indent=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = 2.0
    pf.space_after = Pt(0)
    if indent:
        pf.first_line_indent = INDENT
    add_runs(p, text)
    return p


def caption(doc, text, style="Caption Gambar", space_before=6, space_after=12):
    p = doc.add_paragraph(style=style)
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    add_runs(p, text)
    for r in p.runs:
        r.font.size = Pt(12)
        r.font.name = BODY_FONT
        r.bold = False
        r.italic = False
    return p


def chapter_heading(doc, bab, title, first=False):
    """One Heading 1 paragraph holding both lines, so the Daftar Isi shows a
    single entry ("BAB I PENDAHULUAN") while the page shows two centred lines."""
    if not first:
        doc.add_page_break()
    p = doc.add_paragraph(style="Heading 1")
    r = p.add_run(bab.upper())
    r.bold = True; r.font.size = Pt(14); r.font.name = BODY_FONT
    if title:
        r.add_break()
        r2 = p.add_run(title.upper())
        r2.bold = True; r2.font.size = Pt(14); r2.font.name = BODY_FONT
    return p


def section_heading(doc, text, level=2):
    p = doc.add_paragraph(style=f"Heading {min(level, 3)}")
    pf = p.paragraph_format
    pf.left_indent = Cm(0)
    pf.first_line_indent = Cm(0)
    # "3.2.1 Kebutuhan Data" -> number flush left, title aligned after a tab
    m = re.match(r"^((?:\d+\.)+\d*)\s+(.*)$", text)
    if m:
        num, rest = m.groups()
        pf.tab_stops.add_tab_stop(Cm(1.4 if level == 2 else 1.8))
        parts = [f"{num}\t", rest]
    else:
        parts = [text]
    for part in parts:
        r = p.add_run(part)
        r.bold = True; r.font.size = Pt(12); r.font.name = BODY_FONT
    return p


_MATH_DIR = Path("D:/skripsi/Kodingan/outputs/figures/math")


def render_math(latex: str, index: int):
    """Typeset a display formula to PNG with matplotlib's mathtext."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MATH_DIR.mkdir(parents=True, exist_ok=True)
    path = _MATH_DIR / f"eq_{index:02d}.png"
    body = latex.strip()
    # mathtext understands \frac, \sum, \times; \text{-} and \qquad need help
    body = body.replace(r"\text{-}", "-").replace(r"\qquad", r"\ \ \ \ \ ")
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${body}$", fontsize=16)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06,
                transparent=False, facecolor="white")
    plt.close(fig)
    return path


def add_equation(doc, latex, index):
    path = render_math(latex, index)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    from PIL import Image as _PILImage
    with _PILImage.open(path) as im:
        w_px, h_px = im.size
    width_cm = min(10.0, max(3.2, w_px / 220 * 2.54))
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    return p


def add_figure(doc, md_path, cap_text):
    fname, width_cm = FIGURES[md_path]
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True  # caption must not orphan
    p.add_run().add_picture(str(FIG / fname), width=Cm(width_cm))
    caption(doc, cap_text)


def set_cell_border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "6")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tcPr.append(borders)


def shade_cell(cell, fill="D9D9D9"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def _column_widths(rows, total_cm=14.0):
    """Proportional column widths from content length, with sane bounds."""
    ncol = len(rows[0])
    longest = [1] * ncol
    for row in rows:
        for j, cell in enumerate(row[:ncol]):
            plain = re.sub(r"[*`]", "", cell)
            longest[j] = max(longest[j], min(len(plain), 46))
    total = sum(longest)
    widths = [total_cm * l / total for l in longest]
    # enforce a minimum so narrow columns stay readable, then renormalise
    widths = [max(w, 1.5) for w in widths]
    scale = total_cm / sum(widths)
    return [w * scale for w in widths]


def add_table(doc, rows):
    header, *body = rows
    widths = _column_widths(rows)
    table = doc.add_table(rows=len(rows), cols=len(header))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tblPr = table._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    for j, w in enumerate(widths):
        for cell in table.columns[j].cells:
            cell.width = Cm(w)
    for j, text in enumerate(header):
        cell = table.cell(0, j)
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(2)
        add_runs(p, text)
        for r in p.runs:
            r.bold = True
            r.font.size = Pt(11)
            r.font.name = BODY_FONT
        set_cell_border(cell)
        shade_cell(cell)
    for i, row in enumerate(body, start=1):
        for j, text in enumerate(row):
            if j >= len(header):
                continue
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            # numeric-looking cells centred, text left
            plain = re.sub(r"[*`]", "", text).strip()
            if re.fullmatch(r"[\d.,%\-\u2192\s/()]*", plain):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            add_runs(p, text)
            for r in p.runs:
                r.font.size = Pt(11)
                r.font.name = BODY_FONT
            set_cell_border(cell)
    doc.add_paragraph().paragraph_format.line_spacing = 1.0
    return table


def add_code_block(doc, lines):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.0
    pf.left_indent = Cm(0.6)
    pf.right_indent = Cm(0.2)
    pf.space_before = Pt(6)
    pf.space_after = Pt(12)
    pPr = p._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "6")
        el.set(qn("w:space"), "6")
        el.set(qn("w:color"), "808080")
        borders.append(el)
    pPr.append(borders)
    for k, line in enumerate(lines):
        r = p.add_run(line)
        r.font.name = MONO_FONT
        r.font.size = Pt(9.5)
        if k < len(lines) - 1:
            r.add_break()
    return p


def add_list_item(doc, marker, text, level=0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.line_spacing = 2.0
    pf.space_after = Pt(0)
    pf.left_indent = Cm(1.4 + level * 0.8)
    pf.first_line_indent = Cm(-0.8)
    add_runs(p, f"{marker}\t{text}")
    return p


# ------------------------------------------------------------ markdown parse
CAPTION_FIG_RE = re.compile(r"^\*?(Gambar\s+[\d.]+\s+.*?)\*?$")
CAPTION_TAB_RE = re.compile(r"^(Tabel\s+[\d.]+\s+.*)$")
IMG_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)$")
OL_RE = re.compile(r"^(\d+)\.\s+(.*)$")
UL_RE = re.compile(r"^[-*]\s+(.*)$")


_EQ_COUNTER = {"n": 0}


def parse_markdown(doc, path: Path, first_chapter=False):
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    pending_bab = None
    pending_table_caption = None
    pending_figure = None

    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            i += 1
            continue

        # chapter heading: "# BAB I" followed later by "# PENDAHULUAN"
        if line.startswith("# "):
            text = line[2:].strip()
            if re.match(r"^BAB\s+[IVX]+$", text):
                pending_bab = text
                i += 1
                continue
            if pending_bab:
                chapter_heading(doc, pending_bab, text, first=first_chapter)
                pending_bab = None
                first_chapter = False
            else:
                chapter_heading(doc, text, "", first=first_chapter)
                first_chapter = False
            i += 1
            continue

        if line.startswith("### "):
            section_heading(doc, line[4:].strip(), level=3)
            i += 1
            continue

        if line.startswith("## "):
            section_heading(doc, line[3:].strip(), level=2)
            i += 1
            continue

        # image
        m = IMG_RE.match(line.strip())
        if m:
            pending_figure = m.group(2)
            i += 1
            continue

        # figure caption right after an image
        if pending_figure:
            cap = line.strip().strip("*")
            if cap.startswith("Gambar"):
                add_figure(doc, pending_figure, cap)
                pending_figure = None
                i += 1
                continue
            pending_figure = None

        # table caption -- only when a markdown table actually follows, so that
        # ordinary sentences beginning with "Tabel 4.6 ..." stay body text
        if CAPTION_TAB_RE.match(line.strip().strip("*")):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].lstrip().startswith("|"):
                pending_table_caption = line.strip().strip("*")
                i += 1
                continue

        # markdown table
        if line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                raw = lines[i].strip().strip("|")
                cells = [c.strip() for c in raw.split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if pending_table_caption:
                cap_p = caption(doc, pending_table_caption, style="Caption Tabel",
                                space_before=12, space_after=4)
                cap_p.paragraph_format.keep_with_next = True
                pending_table_caption = None
            if rows:
                add_table(doc, rows)
            continue

        # display equation: $$ ... $$
        if line.strip().startswith("$$"):
            body = line.strip()
            if body.endswith("$$") and len(body) > 4:
                latex = body[2:-2]
                i += 1
            else:
                latex_lines = [body[2:]]
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    latex_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    latex_lines.append(lines[i].strip()[:-2])
                    i += 1
                latex = " ".join(latex_lines)
            _EQ_COUNTER["n"] += 1
            add_equation(doc, latex, _EQ_COUNTER["n"])
            continue

        # fenced code
        if line.strip().startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            add_code_block(doc, code)
            continue

        # horizontal rule
        if set(line.strip()) <= {"-"} and len(line.strip()) >= 3:
            i += 1
            continue

        # ordered list
        m = OL_RE.match(line.strip())
        if m:
            add_list_item(doc, f"{m.group(1)}.", m.group(2))
            i += 1
            continue

        # unordered list
        m = UL_RE.match(line.strip())
        if m:
            add_list_item(doc, "\u2022", m.group(1))
            i += 1
            continue

        body_paragraph(doc, line.strip())
        i += 1


# ------------------------------------------------------------------ assembly
def title_page(doc):
    def line(text, size=14, bold=True, space_after=0, italic=False):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(space_after)
        r = p.add_run(text)
        r.bold = bold; r.italic = italic
        r.font.size = Pt(size); r.font.name = BODY_FONT
        return p

    for _ in range(2):
        doc.add_paragraph().paragraph_format.line_spacing = 1.0
    line("OPTIMASI TRANSFER LEARNING EFFICIENTNET-B0 UNTUK", 14)
    line("KLASIFIKASI KANKER PARU-PARU PADA CITRA COMPUTED", 14)
    line("TOMOGRAPHY (CT) MENGGUNAKAN FINE-TUNING,", 14)
    line("DATA AUGMENTATION, DAN ENSEMBLE MODEL", 14, space_after=36)
    line("SKRIPSI", 12, space_after=36)
    line("diajukan untuk menempuh ujian sarjana", 12, bold=False)
    line("pada Fakultas ...", 12, bold=False)
    line("Universitas ...", 12, bold=False, space_after=48)
    line("[NAMA MAHASISWA]", 12)
    line("NPM [NOMOR POKOK MAHASISWA]", 12, bold=False, space_after=60)
    line("[LOGO UNIVERSITAS]", 11, bold=False, italic=True, space_after=60)
    line("UNIVERSITAS ...", 12)
    line("FAKULTAS ...", 12)
    line("PROGRAM STUDI ...", 12)
    line("[KOTA]", 12)
    line("2026", 12)


def lembar_pengesahan(doc):
    """Approval page: layout follows the standard skripsi template; the names
    are left as fields for the author to fill in."""
    doc.add_page_break()

    def centred(text, size=12, bold=False, after=0, spacing=1.5):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = spacing
        p.paragraph_format.space_after = Pt(after)
        r = p.add_run(text)
        r.bold = bold; r.font.size = Pt(size); r.font.name = BODY_FONT
        return p

    centred("SKRIPSI", 12, True, after=18)
    centred("OPTIMASI TRANSFER LEARNING EFFICIENTNET-B0 UNTUK KLASIFIKASI "
            "KANKER PARU-PARU PADA CITRA COMPUTED TOMOGRAPHY (CT) MENGGUNAKAN "
            "FINE-TUNING, DATA AUGMENTATION, DAN ENSEMBLE MODEL", 12, True, after=18)
    centred("Telah dipersiapkan dan disusun oleh", 12, after=6)
    centred("[NAMA MAHASISWA]", 12, True)
    centred("NPM [NOMOR POKOK MAHASISWA]", 12, after=18)
    centred("Telah dipertahankan di depan Tim Penguji", 12)
    centred("pada tanggal [.....................]", 12, after=18)
    centred("Susunan Tim Penguji", 12, True, after=12)

    rows = [
        ("1.", "[Nama Ketua Tim Penguji]", "Ketua Tim Penguji"),
        ("2.", "[Nama Pembimbing Utama]", "Pembimbing"),
        ("3.", "[Nama Pembimbing Pendamping]", "Co-Pembimbing"),
        ("4.", "[Nama Penguji 1]", "Penguji"),
        ("5.", "[Nama Penguji 2]", "Penguji"),
    ]
    table = doc.add_table(rows=len(rows), cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for j, w in enumerate((0.9, 6.2, 3.8, 3.1)):
        for cell in table.columns[j].cells:
            cell.width = Cm(w)
    for i, (no, nama, peran) in enumerate(rows):
        for j, text in enumerate((no, nama, peran, "....................")):
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.5
            p.paragraph_format.space_after = Pt(10)
            r = p.add_run(text)
            r.font.size = Pt(12); r.font.name = BODY_FONT
            if j == 1:
                r.underline = True


def kata_pengantar(doc):
    simple_heading(doc, "Kata Pengantar")
    paras = [
        "Puji dan syukur penulis panjatkan ke hadirat Tuhan Yang Maha Esa atas "
        "rahmat dan karunia-Nya sehingga penulis dapat menyelesaikan penyusunan "
        "skripsi yang berjudul “OPTIMASI TRANSFER LEARNING EFFICIENTNET-B0 "
        "UNTUK KLASIFIKASI KANKER PARU-PARU PADA CITRA COMPUTED TOMOGRAPHY (CT) "
        "MENGGUNAKAN FINE-TUNING, DATA AUGMENTATION, DAN ENSEMBLE MODEL” "
        "sebagai salah satu syarat menempuh ujian sarjana pada Program Studi "
        "[nama program studi], Fakultas [nama fakultas], Universitas [nama universitas].",
        "Penyusunan skripsi ini tidak terlepas dari bantuan, bimbingan, dan dukungan "
        "berbagai pihak. Oleh karena itu, pada kesempatan ini penulis menyampaikan "
        "terima kasih kepada:",
    ]
    for t in paras:
        body_paragraph(doc, t)
    items = [
        "[Nama Dekan], selaku Dekan Fakultas [nama fakultas], Universitas [nama universitas].",
        "[Nama Ketua Program Studi], selaku Ketua Program Studi [nama program studi].",
        "[Nama Pembimbing Utama], selaku dosen pembimbing utama, atas waktu, arahan, "
        "dan koreksi yang diberikan selama proses penyusunan skripsi ini.",
        "[Nama Pembimbing Pendamping], selaku dosen pembimbing pendamping.",
        "[Nama Penguji], selaku dosen penguji atas masukan yang memperbaiki kualitas "
        "penelitian ini.",
        "Seluruh dosen dan staf Program Studi [nama program studi] yang telah "
        "memberikan ilmu dan bantuan administratif selama masa perkuliahan.",
        "Keluarga penulis yang senantiasa memberikan doa, dukungan, dan semangat.",
        "Rekan-rekan mahasiswa yang telah memberikan bantuan dan diskusi selama "
        "proses penelitian berlangsung.",
    ]
    for k, t in enumerate(items, start=1):
        add_list_item(doc, f"{k}.", t)
    body_paragraph(doc,
                   "Penulis menyadari skripsi ini masih memiliki kekurangan, sehingga "
                   "kritik dan saran yang membangun sangat penulis harapkan. Akhir kata, "
                   "semoga skripsi ini dapat bermanfaat bagi pembaca dan bagi "
                   "pengembangan penelitian selanjutnya.")
    for text, align in (("[Kota], [tanggal]", WD_ALIGN_PARAGRAPH.RIGHT),
                        ("Penulis", WD_ALIGN_PARAGRAPH.RIGHT)):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.line_spacing = 2.0
        r = p.add_run(text)
        r.font.size = Pt(12); r.font.name = BODY_FONT


def daftar_lampiran(doc):
    simple_heading(doc, "Daftar Lampiran")
    entries = [
        "Lampiran 1  Tautan Repositori GitHub Berisi Kode Sumber",
    ]
    for e in entries:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 2.0
        p.paragraph_format.first_line_indent = Cm(0)
        r = p.add_run(e)
        r.font.size = Pt(12); r.font.name = BODY_FONT


def riwayat_hidup(doc):
    doc.add_page_break()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 2.0
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run("RIWAYAT HIDUP")
    r.bold = True; r.font.size = Pt(14); r.font.name = BODY_FONT
    body_paragraph(doc,
                   "[Bagian ini diisi dengan riwayat hidup penulis: nama lengkap, "
                   "tempat dan tanggal lahir, riwayat pendidikan formal dari jenjang "
                   "dasar hingga perguruan tinggi, serta pengalaman organisasi, "
                   "kegiatan, atau prestasi yang relevan.]")


def simple_heading(doc, text, page_break=True):
    if page_break:
        doc.add_page_break()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 2.0
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run(text.upper())
    r.bold = True; r.font.size = Pt(14); r.font.name = BODY_FONT


def add_toc_field(doc, instruction):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar"); fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    sep = OxmlElement("w:fldChar"); sep.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "[Klik kanan di sini lalu pilih Update Field untuk memuat daftar ini]"
    fld_end = OxmlElement("w:fldChar"); fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin); run._r.append(instr); run._r.append(sep)
    run._r.append(placeholder); run._r.append(fld_end)
    run.font.name = BODY_FONT
    run.font.size = BODY_SIZE


def main():
    doc = setup_document()

    # --- front matter (roman numerals) ---
    title_page(doc)
    add_page_number_footer(doc.sections[0], fmt="lowerRoman", start=1)

    front = (SRC / "00_Halaman_Depan_dan_Abstrak.md").read_text(encoding="utf-8")

    lembar_pengesahan(doc)
    kata_pengantar(doc)

    for key, head in (("## ABSTRAK", "Abstrak"), ("## ABSTRACT", "Abstract")):
        simple_heading(doc, head)
        block = front.split(key, 1)[1]
        block = block.split("---", 1)[0]
        for para in [p.strip() for p in block.strip().splitlines() if p.strip()]:
            if para.startswith("**Kata Kunci**") or para.startswith("**Keywords**"):
                body_paragraph(doc, para, indent=False)
            else:
                body_paragraph(doc, para)

    simple_heading(doc, "Daftar Isi")
    add_toc_field(doc, r' TOC \o "1-3" \h \z \u ')
    simple_heading(doc, "Daftar Tabel")
    add_toc_field(doc, r' TOC \h \z \t "Caption Tabel,1" ')
    simple_heading(doc, "Daftar Gambar")
    add_toc_field(doc, r' TOC \h \z \t "Caption Gambar,1" ')
    daftar_lampiran(doc)

    # --- body (arabic numerals, new section) ---
    body_section = doc.add_section(WD_SECTION.NEW_PAGE)
    body_section.page_width = Cm(21.0); body_section.page_height = Cm(29.7)
    body_section.left_margin = Cm(4.0); body_section.right_margin = Cm(3.0)
    body_section.top_margin = Cm(3.0); body_section.bottom_margin = Cm(3.0)
    body_section.footer.is_linked_to_previous = False
    add_page_number_footer(body_section, fmt="decimal", start=1)

    chapters = [
        "BAB_I_Pendahuluan.md",
        "BAB_II_Tinjauan_Pustaka.md",
        "BAB_III_Analisis_dan_Perancangan.md",
        "BAB_IV_Hasil_dan_Pembahasan.md",
        "BAB_V_Kesimpulan_dan_Saran.md",
        "DAFTAR_PUSTAKA.md",
        "LAMPIRAN.md",
    ]
    for k, name in enumerate(chapters):
        parse_markdown(doc, SRC / name, first_chapter=(k == 0))

    riwayat_hidup(doc)

    try:
        doc.save(OUT)
        print(f"Saved: {OUT}", flush=True)
    except PermissionError:
        # dokumen sedang dibuka Word; simpan ke nama cadangan agar hasil tidak hilang
        alt = OUT.with_name(OUT.stem + "_BARU.docx")
        doc.save(alt)
        print(f"Saved: {alt}  (berkas utama terkunci Word)", flush=True)


if __name__ == "__main__":
    main()
