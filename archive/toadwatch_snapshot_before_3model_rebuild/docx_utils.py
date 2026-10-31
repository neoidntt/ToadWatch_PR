"""Small helpers for building the Word report and converting Word / PowerPoint files to PDF."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from . import config as C


def _fig(name: str) -> Path | None:
    p = C.FIGURES_DIR / f"{name}.png"
    return p if p.exists() else None


# ====================================================================== PDF
def to_pdf(path: Path) -> Path | None:
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if not exe:
        print(f"  (LibreOffice not found - skipped PDF for {path.name})")
        return None
    subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(path.parent), str(path)],
                   capture_output=True, timeout=300)
    pdf = path.with_suffix(".pdf")
    return pdf if pdf.exists() else None


# ====================================================================== DOCX
class _Doc:
    def __init__(self, d: dict):
        self.d = d
        self.doc = Document()
        s = self.doc.sections[0]
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(s, m, Inches(1))
        st = self.doc.styles["Normal"]
        st.font.name = "Times New Roman"; st.font.size = Pt(12)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        st.paragraph_format.space_after = Pt(7); st.paragraph_format.line_spacing = 1.5
        for name, sz, it, al in (("Heading 1", 14, False, WD_ALIGN_PARAGRAPH.CENTER),
                                 ("Heading 2", 12, True, WD_ALIGN_PARAGRAPH.LEFT)):
            h = self.doc.styles[name]
            h.font.name = "Times New Roman"; h.font.size = Pt(sz); h.font.bold = True; h.font.italic = it
            h.font.color.rgb = RGBColor(0, 0, 0)
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia"):
                h.element.rPr.rFonts.set(qn(a), "Times New Roman")
            h.paragraph_format.alignment = al
            h.paragraph_format.space_before = Pt(14); h.paragraph_format.space_after = Pt(6)
        fp = s.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = fp.add_run(f"Comparing ML Methods for Detecting TOAD Scams in Puerto Rico · {d['tag']} · generated {d['stamp']} · page ")
        r.font.size = Pt(8)
        r2 = fp.add_run(); r2.font.size = Pt(8)
        for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
            if t:
                e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t)
            else:
                e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
            r2._r.append(e)

    def h1(self, t): self.doc.add_heading(t, 1)
    def h2(self, t): self.doc.add_heading(t, 2)

    def p(self, text="", lead=None, align=None, italic=False, size=None):
        para = self.doc.add_paragraph()
        if lead:
            para.add_run(lead).bold = True
        r = para.add_run(text); r.italic = italic
        if size: r.font.size = Pt(size)
        if align is not None: para.alignment = align
        return para

    def bullets(self, items):
        for t in items:
            q = self.doc.add_paragraph(t, style="List Bullet"); q.paragraph_format.line_spacing = 1.15

    def cap(self, t): self.p(t, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=10)

    def table(self, head, rows, cap=None):
        t = self.doc.add_table(rows=1, cols=len(head)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(head):
            c = t.rows[0].cells[i]; c.text = ""
            r = c.paragraphs[0].add_run(str(h)); r.bold = True; r.font.size = Pt(9)
            sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), "E3E8EF")
            c._tc.get_or_add_tcPr().append(sh)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""; cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(9)
        for row in t.rows:
            for c in row.cells:
                for q in c.paragraphs:
                    q.paragraph_format.line_spacing = 1.0; q.paragraph_format.space_after = Pt(2)
        if cap: self.cap(cap)

    def fig(self, name, caption, width=6.0):
        f = _fig(name)
        if f is None:
            return
        self.doc.add_picture(str(f), width=Inches(width))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.cap(caption)


