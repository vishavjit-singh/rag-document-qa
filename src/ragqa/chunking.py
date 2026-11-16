"""Heading-aware chunking with overlapping word windows, plus content-hash de-duplication.

Chunks never cross a chapter (level-1 heading). Inside a chapter, short sections are packed together up to the
window size; a window prefers to end on a section boundary, then on a paragraph or table-row boundary. Overlap
is only applied inside a section, so a chunk never starts with the tail of an unrelated section.
"""
from __future__ import annotations

import hashlib

from .extract import ExtractedDoc
from .text import content_hash


def _windows(n: int, sections: list[int], blocks: list[int], size: int, overlap: int) -> list[tuple[int, int]]:
    out, start = [], 0
    sec = set(sections)
    while start < n:
        end = min(n, start + size)
        if end < n:
            cut = [b for b in sections if start + 0.5 * size < b <= end] or \
                  [b for b in blocks if start + 0.6 * size < b <= end]
            if cut:
                end = cut[-1]
        out.append((start, end))
        if end >= n:
            break
        if end in sec:
            start = end
            continue
        nxt = max(end - overlap, start + 1)
        snap = [b for b in blocks if nxt <= b < end]
        start = snap[0] if snap else nxt
    return out


def chunk_document(doc: ExtractedDoc, size: int = 120, overlap: int = 30) -> list[dict]:
    """Each chunk keeps the page and section of every line it contains."""
    meta = doc.meta
    title = meta.get("title") or doc.file
    chunks: list[dict] = []
    chapter = "Front matter"
    words: list[tuple[str, int, int, str, bool]] = []      # (word, page, block index, section, is heading)

    def flush():
        if not words:
            return
        n = len(words)
        sec_b = [i for i in range(1, n) if words[i][3] != words[i - 1][3]]
        blk_b = [i for i in range(1, n) if words[i][2] != words[i - 1][2]]
        for start, end in _windows(n, sec_b, blk_b, size, overlap):
            span = words[start:end]
            lines: list[list] = []
            for w, page, blk, sec, head in span:
                if lines and lines[-1][4] == blk:
                    lines[-1][1] += " " + w
                else:
                    lines.append([page, w, sec, head, blk])
            body = "\n".join(t for _, t, _, _, _ in lines)
            secs = list(dict.fromkeys(s for _, _, s, _, _ in lines))
            cid = hashlib.sha1(f"{doc.file}|{chapter}|{start}".encode()).hexdigest()[:12]
            chunks.append({
                "id": cid, "doc": doc.file, "title": title, "code": meta.get("code", ""),
                "version": meta.get("version", ""), "status": meta.get("status", "Current"),
                "effective": meta.get("effective", ""), "chapter": chapter,
                "section": f"{chapter} > {secs[0]}" if len(secs) == 1 and secs[0] != chapter else chapter,
                "sections": secs,
                "pages": sorted({p for _, p, _, _, _ in span}),
                "lines": [[p, t, s, h] for p, t, s, h, _ in lines],
                "text": body, "hash": content_hash(body), "also_in": [],
            })
        words.clear()

    section = chapter
    for bi, b in enumerate(doc.blocks):
        if b.kind == "heading" and b.level == 0:
            continue
        if b.kind == "heading" and b.level == 1:
            flush()
            chapter = section = b.text
            continue
        if b.kind == "heading":
            section = b.text
        words.extend((w, b.page, bi, section, b.kind == "heading") for w in b.text.split())
    flush()
    return chunks


def embed_text(chunk: dict) -> str:
    """What the retrievers see: document title, version and chapter, then the chunk body (with headings)."""
    return f"{chunk['title']} {chunk['version']} > {chunk['chapter']}\n{chunk['text']}"


def dedupe(chunks: list[dict]) -> tuple[list[dict], int]:
    """Collapse chunks with identical normalised text. The copy from a current document is kept as the
    primary and the other locations are listed in ``also_in`` so citations and evaluation can still use them."""
    groups: dict[str, list[dict]] = {}
    for c in chunks:
        groups.setdefault(c["hash"], []).append(c)
    out, dropped = [], 0
    for c in chunks:
        g = groups[c["hash"]]
        primary = next((x for x in g if x["status"] != "Superseded"), g[0])
        if c is not primary:
            dropped += 1
            continue
        c["also_in"] = [{"doc": x["doc"], "pages": x["pages"]} for x in g if x is not primary]
        out.append(c)
    return out, dropped
