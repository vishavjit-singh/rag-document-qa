"""Render the SYNTHETIC corpus to PDF with reportlab and write the gold QA set.

The gold page of every fact is recorded by the layout engine (``afterFlowable``) while the PDF is built,
so the evaluation does not depend on the extraction code it is evaluating.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

from .spec import COMPANY, DOCS, QAS, SYNTHETIC_NOTE, DocSpec

log = logging.getLogger(__name__)
rl_config.invariant = 1          # no timestamps or random IDs in the PDF bytes: same input, same file hash

INK = colors.HexColor("#1d2a3a")
MUTED = colors.HexColor("#5b6676")
RULE = colors.HexColor("#c9d1dc")
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=14.5, textColor=INK, spaceAfter=7,
                      alignment=TA_LEFT)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=INK, spaceBefore=4,
                    spaceAfter=10)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=INK, spaceBefore=8,
                    spaceAfter=5)
TITLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=24, leading=30, textColor=INK, spaceAfter=8)
CELL = ParagraphStyle("cell", fontName="Helvetica", fontSize=9.5, leading=12, textColor=INK)
CELL_B = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")


class _Doc(BaseDocTemplate):
    """Records the page number of every flowable tagged with ``_fact``."""

    def __init__(self, path: str, spec: DocSpec, **kw):
        super().__init__(path, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=24 * mm,
                         bottomMargin=22 * mm, title=f"{spec.title} {spec.version}", author=f"{COMPANY} (fictional)",
                         subject="SYNTHETIC demo document", creator="ragqa.synth",
                         keywords=f"code={spec.code}; version={spec.version}; status={spec.status}", **kw)
        self.spec = spec
        self.fact_pages: dict[str, int] = {}
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=self._decorate)])

    def afterFlowable(self, flowable):
        for key in getattr(flowable, "_facts", ()):
            self.fact_pages.setdefault(key, self.page)

    def _decorate(self, canvas, doc):
        s = self.spec
        w, h = A4
        canvas.saveState()
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(22 * mm, h - 14 * mm, f"{COMPANY} (fictional) | {s.title}")
        canvas.drawRightString(w - 22 * mm, h - 14 * mm, f"{s.code} v{s.version}")
        canvas.setStrokeColor(RULE)
        canvas.line(22 * mm, h - 16 * mm, w - 22 * mm, h - 16 * mm)
        canvas.line(22 * mm, 16 * mm, w - 22 * mm, 16 * mm)
        canvas.drawString(22 * mm, 11 * mm, "SYNTHETIC document generated for a portfolio demo")
        canvas.drawRightString(w - 22 * mm, 11 * mm, f"Page {doc.page}")
        canvas.restoreState()


def _tag(flowable, *keys):
    flowable._facts = tuple(k for k in keys if k)
    return flowable


def _table(header: list[str], rows: list[list[str]], widths=None) -> Table:
    data = [[Paragraph(c, CELL_B) for c in header]] + [[Paragraph(c, CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.6, RULE),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _title_page(spec: DocSpec) -> list:
    rows = [["Document code", spec.code], ["Version", spec.version], ["Effective date", spec.effective],
            ["Status", spec.status], ["Owner", spec.owner], ["Supersedes", spec.supersedes]]
    if spec.superseded_by:
        rows.append(["Superseded by", spec.superseded_by])
    out = [Spacer(1, 30 * mm), Paragraph(spec.title, TITLE),
           Paragraph(f"{COMPANY} (fictional company)", ParagraphStyle("sub", parent=BODY, fontSize=13, leading=17,
                                                                      textColor=MUTED, spaceAfter=18)),
           Paragraph("Document control", H2), _table(["Field", "Value"], rows, widths=[45 * mm, 80 * mm]),
           Spacer(1, 10 * mm)]
    if spec.status == "Superseded":
        out.append(Paragraph(f"This version is superseded by version {spec.superseded_by} and is kept for "
                             f"reference only.", ParagraphStyle("warn", parent=BODY, textColor=colors.HexColor(
                                 "#b42318"), fontName="Helvetica-Bold")))
    out.append(Paragraph(SYNTHETIC_NOTE, ParagraphStyle("note", parent=BODY, textColor=MUTED, fontSize=9.5)))
    return out


def _story(spec: DocSpec) -> list:
    story = _title_page(spec)
    for ci, ch in enumerate(spec.chapters, 1):
        story += [PageBreak(), Paragraph(f"{ci}. {ch.title}", H1)]
        for si, sec in enumerate(ch.sections, 1):
            story.append(Paragraph(f"{ci}.{si} {sec.heading}", H2))
            for b in sec.blocks:
                if b.kind == "p":
                    story.append(KeepTogether([_tag(Paragraph(b.text, BODY), b.fact)]))
                elif b.kind == "bullets":
                    story.append(KeepTogether([_tag(Paragraph(i, BULLET, bulletText="•"), b.fact)
                                               for i in b.items]))
                else:
                    t = _tag(_table(b.header, b.rows), b.fact, *b.row_facts.values())
                    story += [KeepTogether([t]), Spacer(1, 6)]
    return story


def fact_texts() -> dict[str, tuple[str, str]]:
    """fact key -> (file, text of the block that holds it)."""
    out = {}
    for d in DOCS:
        for ch in d.chapters:
            for sec in ch.sections:
                for b in sec.blocks:
                    if b.fact:
                        txt = b.text if b.kind == "p" else " ".join(b.items) if b.kind == "bullets" \
                            else " ".join(" ".join(r) for r in b.rows)
                        out[b.fact] = (d.file, txt)
                    for i, key in b.row_facts.items():
                        out[key] = (d.file, " ".join(b.rows[i]))
    return out


def build_corpus(corpus_dir: Path, gold_path: Path) -> dict:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    pages: dict[str, int] = {}
    fact_page: dict[str, int] = {}
    for spec in DOCS:
        doc = _Doc(str(corpus_dir / spec.file), spec)
        doc.build(_story(spec))
        pages[spec.file] = doc.page
        fact_page.update(doc.fact_pages)
        log.info("wrote %s (%d pages)", spec.file, doc.page)

    texts = fact_texts()
    status = {d.file: d.status for d in DOCS}
    rows = []
    for q in QAS:
        if q.fact is None:
            rows.append({"id": q.id, "question": q.question, "answerable": False})
            continue
        file, text = texts[q.fact]
        if q.fact not in fact_page:
            raise RuntimeError(f"fact {q.fact} was not placed on a page")
        if q.answer not in text:
            raise RuntimeError(f"answer span {q.answer!r} not found in fact {q.fact}")
        rows.append({"id": q.id, "question": q.question, "answerable": True, "doc": file, "page": fact_page[q.fact],
                     "answer": q.answer, "fact": q.fact, "superseded_doc": status[file] == "Superseded"})
    with gold_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {"documents": len(DOCS), "pages": sum(pages.values()), "pages_per_doc": pages, "questions": len(rows),
            "answerable": sum(r["answerable"] for r in rows),
            "unanswerable": sum(not r["answerable"] for r in rows)}
