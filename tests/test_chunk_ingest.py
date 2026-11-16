import json
import shutil

from ragqa import store
from ragqa.chunking import _windows, chunk_document, dedupe
from ragqa.config import with_overrides
from ragqa.extract import extract
from tests.conftest import make_pdf


def test_windows_respect_size_and_prefer_boundaries():
    w = _windows(100, sections=[50], blocks=[20, 35, 50, 70, 90], size=40, overlap=10)
    assert all(e - s <= 40 for s, e in w)
    assert (0, 35) in w                     # ended on a block boundary inside the window
    assert any(e == 50 for _, e in w) and any(s == 50 for s, _ in w)   # no overlap across the section break
    assert w[-1][1] == 100


def test_chunks_stay_inside_a_chapter_and_keep_line_pages(cfg, gold):
    d = extract(cfg["paths"]["corpus"] / "employee-handbook-2025.pdf")
    chunks = chunk_document(d, size=40, overlap=10)
    assert max(len(c["text"].split()) for c in chunks) <= 40
    for c in chunks:
        assert c["section"].startswith(c["chapter"])
        assert c["pages"] == sorted({p for p, *_ in c["lines"]})
    q = next(q for q in gold if q["id"] == "hr09")      # parental leave, 26 weeks
    line = next(ln for c in chunks for ln in c["lines"] if q["answer"] in ln[1])
    assert line[0] == q["page"] and "Parental leave" in line[2]


def test_dedup_keeps_the_current_copy():
    base = {"title": "T", "code": "C", "version": "", "effective": "", "section": "s", "chapter": "c", "lines": [],
            "pages": [2], "also_in": []}
    old = {**base, "id": "1", "doc": "old.pdf", "status": "Superseded", "hash": "h"}
    new = {**base, "id": "2", "doc": "new.pdf", "status": "Current", "hash": "h", "pages": [3]}
    other = {**base, "id": "3", "doc": "x.pdf", "status": "Current", "hash": "k"}
    out, dropped = dedupe([old, new, other])
    assert dropped == 1 and [c["id"] for c in out] == ["2", "3"]
    assert out[0]["also_in"] == [{"doc": "old.pdf", "pages": [2]}]


def test_ingest_is_incremental(fresh):
    c = fresh["paths"]
    s = store.ingest(c["corpus"], c["index"], fresh)
    assert s["up_to_date"] and s["parsed"] == 0
    make_pdf(c["corpus"] / "canteen-notice.pdf", "Canteen notice", "The canteen opens at 08:00 on weekdays.")
    shutil.copy(c["corpus"] / "track-sla-2025.pdf", c["corpus"] / "track-sla-copy.pdf")
    (c["corpus"] / "freight-claims-returns-policy.pdf").unlink()
    s = store.ingest(c["corpus"], c["index"], fresh)
    assert s["added"] == ["canteen-notice.pdf", "track-sla-copy.pdf"]
    assert s["removed"] == ["freight-claims-returns-policy.pdf"] and s["changed"] == []
    assert s["parsed"] == 1             # the copy has the same bytes, so its cached extraction is reused
    assert s["duplicates_dropped"] >= 5                       # and its chunks are exact duplicates
    files = json.loads((c["index"] / "manifest.json").read_text())["files"]
    assert "freight-claims-returns-policy.pdf" not in files and "canteen-notice.pdf" in files
    make_pdf(c["corpus"] / "canteen-notice.pdf", "Canteen notice", "The canteen opens at 07:30 on weekdays.")
    s = store.ingest(c["corpus"], c["index"], fresh)
    assert s["changed"] == ["canteen-notice.pdf"] and s["parsed"] == 1 and s["unchanged"] == 9


def test_settings_change_rechunks_without_reparsing(fresh):
    c = fresh["paths"]
    smaller = with_overrides(fresh, chunking={"size": 50, "overlap": 10})
    s = store.ingest(c["corpus"], c["index"], smaller)
    assert not s["up_to_date"] and s["parsed"] == 0 and s["chunks"] > 56


def test_stateless_embedder_reuses_vectors(fresh):
    c = fresh["paths"]
    hashing = with_overrides(fresh, embedding={"backend": "hashing"})
    first = store.ingest(c["corpus"], c["index"], hashing)
    assert first["chunks_embedded"] == first["chunks"]
    make_pdf(c["corpus"] / "parking.pdf", "Parking rules", "Visitor parking is in bay C behind Gate 2.")
    second = store.ingest(c["corpus"], c["index"], hashing)
    assert second["parsed"] == 1 and second["chunks"] == first["chunks"] + 1
    assert second["chunks_embedded"] == 1                    # only the new chunk; the rest kept their vectors
