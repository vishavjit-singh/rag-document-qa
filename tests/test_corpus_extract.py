import hashlib
import re

from pypdf import PdfReader

from ragqa.extract import clean_lines, extract, linearise_table
from ragqa.synth.build import build_corpus
from ragqa.synth.spec import DOCS, QAS
from ragqa.text import normalise


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_corpus_is_deterministic(cfg, tmp_path):
    stats = build_corpus(tmp_path / "corpus", tmp_path / "qa.jsonl")
    assert stats["documents"] == len(DOCS) and stats["questions"] == len(QAS)
    for p in sorted(cfg["paths"]["corpus"].glob("*.pdf")):
        assert _sha(p) == _sha(tmp_path / "corpus" / p.name), p.name


def test_gold_spans_are_on_the_recorded_page(cfg, gold):
    """The gold page comes from the layout engine; check it independently with pypdf."""
    readers = {}
    answerable = [q for q in gold if q["answerable"]]
    assert len(answerable) >= 60 and sum(not q["answerable"] for q in gold) >= 10
    for q in answerable:
        r = readers.setdefault(q["doc"], PdfReader(str(cfg["paths"]["corpus"] / q["doc"])))
        page_text = normalise(r.pages[q["page"] - 1].extract_text())
        assert normalise(q["answer"]) in page_text, (q["id"], q["page"])


def test_every_page_is_labelled_synthetic(cfg):
    for p in cfg["paths"]["corpus"].glob("*.pdf"):
        for page in PdfReader(str(p)).pages:
            assert "SYNTHETIC" in page.extract_text()


def test_pdfplumber_strips_running_headers_and_footers(cfg):
    d = extract(cfg["paths"]["corpus"] / "track-sla-2025.pdf")
    text = "\n".join(b.text for b in d.blocks)
    assert d.stripped_lines == 2 * d.pages
    assert not re.search(r"Page \d", text) and "KL-LEG-007 v2.0" not in text
    assert "portfolio demo" not in text.replace("SYNTHETIC DOCUMENT. Kestrelway Logistics is a fictional company; "
                                                "this file was generated for a portfolio demo", "")


def test_tables_are_linearised_with_column_names(cfg):
    d = extract(cfg["paths"]["corpus"] / "travel-expense-policy-2025.pdf")
    rows = [b for b in d.blocks if b.kind == "table_row"]
    assert any(b.text == "Tier 1 - Hotel cap per night: INR 7,500; Meal allowance per day: INR 2,000"
               and b.page == 4 for b in rows)
    assert any(b.text == "Status: Current" and b.page == 1 for b in rows)


def test_headings_levels_and_document_metadata(cfg):
    d = extract(cfg["paths"]["corpus"] / "travel-expense-policy-2023.pdf")
    heads = {(b.text, b.level) for b in d.blocks if b.kind == "heading"}
    assert ("Travel and Expense Policy", 0) in heads
    assert ("2. Approval and booking", 1) in heads and ("2.2 Air travel", 2) in heads
    assert d.meta == {"title": "Travel and Expense Policy", "code": "KL-FIN-004", "version": "2023.2",
                      "effective": "1 July 2023", "status": "Superseded", "superseded_by": "2025.1"}


def test_pypdf_fallback_gets_the_same_metadata_and_text(cfg):
    a = extract(cfg["paths"]["corpus"] / "track-sla-2024.pdf", "pdfplumber")
    b = extract(cfg["paths"]["corpus"] / "track-sla-2024.pdf", "pypdf")
    for k in ("title", "code", "version", "status"):
        assert a.meta[k] == b.meta[k]
    text = " ".join(x.text for x in b.blocks)
    assert "at least 99.5% of the time" in text and "Page 3" not in text


def test_cleaning_helpers():
    assert clean_lines(["• Use the travel desk for book-", "ings made abroad.", "Next   line"]) == \
        ["Use the travel desk for bookings made abroad.", "Next line"]
    assert linearise_table([["LED", "Meaning"], ["Red, steady", "Hardware\nfault"]]) == ["Red, steady: Hardware fault"]
    assert linearise_table([["P", "A", "B"], ["P1", "x", None]]) == ["P1 - A: x"]
