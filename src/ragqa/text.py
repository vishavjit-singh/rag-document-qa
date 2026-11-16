"""Tokenising, light stemming and sentence splitting shared by BM25, the embedders and the answerer."""
from __future__ import annotations

import hashlib
import re
import unicodedata

STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does doing for from had has have having how i if in into
is it its may me might must my of on or our shall should so than that the their them then there these they this
those to under up us was we were what when where which while who whom why will with would you your yes no not
am any all also about after before over per each many much get gets got
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*%?")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def normalise(text: str) -> str:
    """NFKC, unify dashes and quotes, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def stem(tok: str) -> str:
    """A deliberately small suffix stripper: enough to match 'claims'/'claim', 'days'/'day', 'booked'/'book'."""
    if tok[0].isdigit() or len(tok) <= 3:
        return tok
    for suf, rep, min_len in (("ies", "y", 5), ("sses", "ss", 5), ("ing", "", 6), ("ed", "", 5)):
        if tok.endswith(suf) and len(tok) >= min_len:
            return tok[: -len(suf)] + rep
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokens(text: str, drop_stopwords: bool = True) -> list[str]:
    out = []
    for t in _TOKEN.findall(normalise(text).lower()):
        t = t.rstrip(".,")
        if not t or (drop_stopwords and t in STOPWORDS):
            continue
        out.append(stem(t))
    return out


def sentences(text: str) -> list[str]:
    """Split a block of text into sentences; short fragments are attached to the previous sentence."""
    out: list[str] = []
    for part in _SENT.split(text.strip()):
        part = part.strip()
        if not part:
            continue
        if out and len(part.split()) < 3:
            out[-1] += " " + part
        else:
            out.append(part)
    return out


def content_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).lower().encode()).hexdigest()[:16]


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
