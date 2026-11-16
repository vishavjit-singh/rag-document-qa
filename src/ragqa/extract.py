"""PDF text extraction: one list of typed blocks per document.

Default engine is pdfplumber (layout-aware): lines are grouped into paragraphs by vertical gaps, headings are
found by font size, tables are detected and linearised row by row with their column names, and running
headers/footers are dropped when the same line (digits ignored) repeats in the top or bottom band of most
pages. The pypdf engine is a plain-text fallback: headings by numbering pattern, no table structure.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .text import file_hash, normalise

HEADER_BAND, FOOTER_BAND, REPEAT_SHARE = 0.09, 0.09, 0.6
_NUMBERED_HEADING = re.compile(r"^(\d+)(?:\.(\d+))?\.?\s+([A-Z][^.!?]{2,80})$")
_BULLET = re.compile(r"^[•●▪\-\*]\s+")


@dataclass
class TextBlock:
    kind: str                 # "heading" | "text" | "table_row"
    text: str
    page: int                 # 1-based
    level: int = 0            # heading level: 0 = document title, 1 = chapter, 2 = section


@dataclass
class ExtractedDoc:
    file: str
    sha256: str
    pages: int
    engine: str
    meta: dict
    blocks: list[TextBlock] = field(default_factory=list)
    stripped_lines: int = 0   # header/footer lines removed

    def to_dict(self) -> dict:
        return {**asdict(self), "blocks": [asdict(b) for b in self.blocks]}

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractedDoc":
        return cls(**{**d, "blocks": [TextBlock(**b) for b in d["blocks"]]})


def _norm_key(line: str) -> str:
    return re.sub(r"\d+", "#", normalise(line).lower())


def clean_lines(lines: list[str]) -> list[str]:
    """Join words hyphenated across line ends and drop bullet glyphs."""
    out: list[str] = []
    for ln in lines:
        ln = _BULLET.sub("", normalise(ln))
        if out and out[-1].endswith("-") and ln[:1].islower() and not out[-1].endswith(" -"):
            out[-1] = out[-1][:-1] + ln
        elif ln:
            out.append(ln)
    return out


def linearise_table(rows: list[list[str | None]]) -> list[str]:
    """Each data row becomes one self-contained line: 'Tier 1 - Hotel cap per night: INR 7,500; ...'."""
    rows = [[normalise(c or "") for c in r] for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return [" ".join(r) for r in rows]
    header, body = rows[0], rows[1:]
    out = []
    for r in body:
        if len(r) == 2:
            out.append(f"{r[0]}: {r[1]}")
        else:
            pairs = "; ".join(f"{h}: {v}" for h, v in zip(header[1:], r[1:]) if v)
            out.append(f"{r[0]} - {pairs}")
    return out


def _repeated(keys_per_page: list[set[str]]) -> set[str]:
    if len(keys_per_page) < 2:
        return set()
    counts = Counter(k for ks in keys_per_page for k in ks)
    return {k for k, c in counts.items() if c >= max(2, REPEAT_SHARE * len(keys_per_page))}


def parse_meta(blocks: list[TextBlock], pdf_meta: dict | None = None) -> dict:
    """Document control fields from the first page, falling back to the PDF's own metadata."""
    first = " \n".join(b.text for b in blocks if b.page == 1)
    meta = {"title": next((b.text for b in blocks if b.kind == "heading" and b.level == 0), "")}
    for key, label in (("code", "Document code"), ("version", "Version"), ("effective", "Effective date"),
                       ("status", "Status"), ("superseded_by", "Superseded by")):
        m = re.search(rf"{label}:?\s+([^\n;]+?)\s*(?:\n|$)", first)
        if m:
            meta[key] = m.group(1).strip()
    kw = (pdf_meta or {}).get("/Keywords", "") or ""
    for k, v in re.findall(r"(\w+)=([^;]+)", kw):
        meta.setdefault(k, v.strip())
    if not meta["title"]:
        meta["title"] = re.sub(r"\s+\d[\w.]*$", "", (pdf_meta or {}).get("/Title", "") or "")
    meta.setdefault("status", "Current")
    return meta


# ------------------------------------------------------------------------------------------- pdfplumber
def _extract_pdfplumber(path: Path) -> tuple[list[TextBlock], int, int, dict]:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        pdf_meta = {f"/{k}": v for k, v in (pdf.metadata or {}).items()}
        raw_pages = []
        sizes: Counter = Counter()
        for page in pdf.pages:
            tables = page.find_tables()
            boxes = [t.bbox for t in tables]
            lines = []
            for ln in page.extract_text_lines(return_chars=True):
                cx = (ln["x0"] + ln["x1"]) / 2
                if any(b[0] <= cx <= b[2] and b[1] <= ln["top"] <= b[3] for b in boxes):
                    continue
                size = round(max(c["size"] for c in ln["chars"]), 1)
                bold = all("Bold" in c["fontname"] for c in ln["chars"] if c["text"].strip())
                lines.append({"top": ln["top"], "bottom": ln["bottom"], "text": ln["text"], "size": size,
                              "bold": bold})
                sizes[size] += len(ln["text"])
            raw_pages.append((page.height, lines, [(t.bbox[1], t.extract()) for t in tables]))

    body = sizes.most_common(1)[0][0] if sizes else 10.0
    band_keys = []
    for height, lines, _ in raw_pages:
        band_keys.append({_norm_key(ln["text"]) for ln in lines
                          if ln["top"] < HEADER_BAND * height or ln["bottom"] > (1 - FOOTER_BAND) * height})
    repeated = _repeated(band_keys)
    bold_sizes = sorted({ln["size"] for _, ls, _ in raw_pages for ln in ls if ln["bold"] and ln["size"] > body + 0.9},
                        reverse=True)

    def heading_level(size: float) -> int:      # largest bold size = document title when there are 3+ levels
        i = bold_sizes.index(size)
        return min(i, 2) if len(bold_sizes) >= 3 else min(i + 1, 2)

    blocks: list[TextBlock] = []
    stripped = 0
    for pno, (height, lines, tables) in enumerate(raw_pages, 1):
        kept = []
        for ln in lines:
            in_band = ln["top"] < HEADER_BAND * height or ln["bottom"] > (1 - FOOTER_BAND) * height
            if in_band and _norm_key(ln["text"]) in repeated:
                stripped += 1
                continue
            kept.append(ln)
        items = [(ln["top"], "line", ln) for ln in kept] + [(top, "table", rows) for top, rows in tables]
        para: list[str] = []
        prev = None

        def flush():
            if para:
                blocks.append(TextBlock("text", " ".join(clean_lines(para)), pno))
                para.clear()

        for _, kind, item in sorted(items, key=lambda x: x[0]):
            if kind == "table":
                flush()
                blocks += [TextBlock("table_row", r, pno) for r in linearise_table(item)]
                prev = None
                continue
            ln = item
            if ln["bold"] and ln["size"] > body + 0.9:
                flush()
                level, text = heading_level(ln["size"]), normalise(ln["text"])
                if blocks and blocks[-1].kind == "heading" and prev is not None and prev["size"] == ln["size"] \
                        and ln["top"] - prev["bottom"] < 0.6 * ln["size"]:
                    blocks[-1].text += " " + text                 # heading wrapped onto a second line
                else:
                    blocks.append(TextBlock("heading", text, pno, level))
                prev = ln
                continue
            gap = ln["top"] - prev["bottom"] if prev is not None else 0
            if prev is not None and (gap > 0.7 * ln["size"] or prev["size"] != ln["size"]):
                flush()
            para.append(ln["text"])
            prev = ln
        flush()
    return blocks, len(raw_pages), stripped, pdf_meta


# ------------------------------------------------------------------------------------------------- pypdf
def _extract_pypdf(path: Path) -> tuple[list[TextBlock], int, int, dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pdf_meta = {k: str(v) for k, v in (reader.metadata or {}).items()}
    pages = [[ln for ln in (p.extract_text() or "").splitlines() if ln.strip()] for p in reader.pages]
    # pypdf emits lines in drawing order, so running headers/footers can sit anywhere in the page text:
    # drop short lines that repeat (digits ignored) on most pages
    repeated = _repeated([{_norm_key(x) for x in ls if len(x.split()) <= 15} for ls in pages])
    title = re.sub(r"\s+v?\d[\w.]*$", "", pdf_meta.get("/Title", "")).strip()
    blocks: list[TextBlock] = [TextBlock("heading", title, 1, 0)] if title else []
    stripped = 0
    for pno, lines in enumerate(pages, 1):
        para: list[str] = []
        for ln in lines:
            if _norm_key(ln) in repeated:
                stripped += 1
                continue
            m = _NUMBERED_HEADING.match(normalise(ln)) if pno > 1 else None
            if m or pno == 1:            # title page: keep lines separate so "Status / Current" pairs survive
                if para:
                    blocks.append(TextBlock("text", " ".join(clean_lines(para)), pno))
                    para = []
                blocks.append(TextBlock("heading", normalise(ln), pno, 2 if m.group(2) else 1) if m
                              else TextBlock("text", normalise(ln), pno))
            else:
                para.append(ln)
        if para:
            blocks.append(TextBlock("text", " ".join(clean_lines(para)), pno))
    return blocks, len(pages), stripped, pdf_meta


ENGINES = {"pdfplumber": _extract_pdfplumber, "pypdf": _extract_pypdf}


def extract(path: str | Path, engine: str = "pdfplumber") -> ExtractedDoc:
    path = Path(path)
    blocks, n_pages, stripped, pdf_meta = ENGINES[engine](path)
    blocks = [b for b in blocks if b.text]
    return ExtractedDoc(file=path.name, sha256=file_hash(path), pages=n_pages, engine=engine,
                        meta=parse_meta(blocks, pdf_meta), blocks=blocks, stripped_lines=stripped)
