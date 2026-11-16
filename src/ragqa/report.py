"""README images, rendered from reports/eval.json and reports/sample_answers.json written by `eval`."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED, GRID = "#0e1a2b", "#5b6676", "#e3e7ee"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]     # fixed categorical order, validated for CVD
GOOD, BAD, PURPLE = "#1baf7a", "#e34948", "#6250d6"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def _save(fig, out: Path, name: str) -> Path:
    fig.tight_layout()
    p = out / name
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def architecture(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white",
                                    ec=color, lw=2))
        ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center", fontsize=10.5, weight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.33, sub, ha="center", va="center", fontsize=8, color=MUTED, linespacing=1.4)

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.5, color=color))

    ax.text(0.15, 6.1, "rag-document-qa: PDFs → cited answers, or an honest \"not found\"", fontsize=13,
            weight="bold", color=INK)
    ax.text(0.2, 5.6, "ingest  (incremental: only new or changed files are parsed, keyed by SHA-256)", fontsize=8.8,
            color=PURPLE)
    steps = [("PDFs", "data/corpus/*.pdf\nSYNTHETIC policies,\nmanual, SLAs"),
             ("Extract", "pdfplumber: paragraphs,\nheadings, tables → rows\nheader/footer stripped"),
             ("Chunk", "word windows + overlap\nnever across chapters\npage kept per line"),
             ("De-duplicate", "content hash; copy\nfrom current version\nkept, others noted"),
             ("Embed + index", "LSA (TF-IDF + SVD)\nFAISS or NumPy\n+ BM25 postings")]
    for i, (t, s) in enumerate(steps):
        box(0.2 + i * 2.55, 3.75, 2.2, 1.55, t, s, PURPLE)
        if i:
            arrow(0.2 + i * 2.55 - 0.35, 4.52, 0.2 + i * 2.55, 4.52)
    ax.text(0.2, 3.2, "ask", fontsize=8.8, color=SERIES[0], weight="bold")
    box(0.2, 0.9, 2.2, 1.9, "Question", "POST /ask\npython -m ragqa ask", SERIES[0])
    box(2.75, 0.9, 2.6, 1.9, "Hybrid retrieval", "BM25 + dense (top 30 each)\nreciprocal rank fusion\n"
        "superseded versions\nranked after current", SERIES[0])
    box(5.7, 0.9, 2.6, 1.9, "Extractive answerer", "score sentences + table rows\nidf coverage, similarity,\n"
        "answer type · rank prior", SERIES[0])
    box(8.65, 1.95, 4.0, 1.0, "Cited answer", "answer [file p.N] · confidence · evidence", GOOD)
    box(8.65, 0.7, 4.0, 1.0, "Abstain", "score < threshold → \"Not found in the documents.\"", BAD)
    arrow(2.4, 1.85, 2.75, 1.85, SERIES[0])
    arrow(5.35, 1.85, 5.7, 1.85, SERIES[0])
    arrow(8.3, 2.3, 8.65, 2.42, GOOD)
    arrow(8.3, 1.4, 8.65, 1.22, BAD)
    arrow(11.4, 3.75, 4.3, 2.8, MUTED)
    ax.text(5.2, 3.38, "chunks, vectors, BM25 postings", fontsize=7.8, color=MUTED)
    ax.text(0.2, 0.35, "eval: gold QA set (doc + page + answer span) → recall@k, MRR, nDCG, answer accuracy, "
            "abstention; "
            "the threshold is chosen on a dev split.   Optional: OpenAI answerer (citations enforced), "
            "sentence-transformers.", fontsize=7.8, color=MUTED)
    return _save(fig, out, "architecture.png")


def retrieval_comparison(out: Path, ev: dict) -> Path:
    r = ev["retrieval"]
    names = list(r)
    metrics = [("recall@1", "Recall@1"), ("recall@5", "Recall@5"), ("mrr@10", "MRR@10"), ("ndcg@10", "nDCG@10")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2.3, 1]})
    w = 0.8 / len(names)
    for j, n in enumerate(names):
        xs = [i + (j - (len(names) - 1) / 2) * w for i in range(len(metrics))]
        vals = [r[n][m] for m, _ in metrics]
        a1.bar(xs, vals, width=w - 0.02, color=SERIES[j], label=n)
        for x, v in zip(xs, vals):
            a1.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, color=INK, rotation=90)
    a1.set_xticks(range(len(metrics)), [lbl for _, lbl in metrics])
    a1.set_ylim(0, 1.13)
    a1.grid(axis="y", color=GRID)
    a1.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.08))
    a1.set_title(f"All {ev['gold']['answerable']} answerable questions")
    vq = [r[n]["version_questions_recall@1"] for n in names]
    a2.barh(range(len(names)), vq, color=SERIES[: len(names)], height=0.6)
    for i, v in enumerate(vq):
        a2.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9, color=INK)
    a2.set_yticks(range(len(names)), names, fontsize=8.5)
    a2.invert_yaxis()
    a2.set_xlim(0, 1.15)
    a2.grid(axis="x", color=GRID)
    a2.set_title(f"Recall@1, {ev['gold']['version_questions']} questions on\ndocuments with two versions")
    best = max(names, key=lambda n: r[n]["mrr@10"])
    fig.suptitle(f"{best} ranks best (MRR@10 {r[best]['mrr@10']:.3f}); lexical retrieval alone often returns "
                 f"the superseded version", x=0.01, ha="left", fontsize=12.5, weight="bold", color=INK)
    return _save(fig, out, "retrieval_comparison.png")


def recall_at_k(out: Path, ev: dict) -> Path:
    r = ev["retrieval"]
    ks = [int(k.split("@")[1]) for k in r[next(iter(r))] if k.startswith("recall@")]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for j, (n, m) in enumerate(r.items()):
        ys = [m[f"recall@{k}"] for k in ks]
        ax.plot(ks, ys, marker="o", ms=6, lw=2, color=SERIES[j], label=n)
    ax.set_xticks(ks)
    ax.set_xlabel("k (chunks returned)")
    ax.set_ylabel("share of questions with the gold page in the top k")
    ax.set_ylim(0.5, 1.02)
    ax.grid(color=GRID)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    h = r["Hybrid + version-aware"]
    ax.set_title(f"Recall@k: by k=5 every method finds the page for {h['recall@5']:.0%} of questions; "
                 f"the methods differ at k=1")
    return _save(fig, out, "recall_at_k.png")


def chunk_size(out: Path, ev: dict) -> Path:
    s = ev["chunk_sweep"]
    xs = [x["size"] for x in s]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for j, (key, label) in enumerate([("recall@1", "Recall@1"), ("recall@5", "Recall@5"),
                                      ("span_accuracy_no_abstention", "answer contains the gold span")]):
        ys = [x[key] for x in s]
        ax.plot(xs, ys, marker="o", ms=6, lw=2, color=SERIES[j], label=label)
    for x in s:
        ax.text(x["size"], 0.62, f"{x['chunks']}\nchunks", ha="center", fontsize=7.5, color=MUTED)
    default = ev["settings"]["chunking"]["size"]
    ax.vlines(default, 0.69, 1.03, color=MUTED, ls="--", lw=1)
    ax.text(default + 3, 1.0, f"default {default}", fontsize=8, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xlabel("chunk size (words; overlap = 25%)")
    ax.set_ylim(0.58, 1.03)
    ax.grid(color=GRID)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1, 0.2))
    best = max(s, key=lambda x: (x["recall@1"], -x["size"]))
    ax.set_title(f"Chunk size: smaller windows put the right page first more often (best Recall@1 "
                 f"{best['recall@1']:.2f} at {best['size']} words)")
    return _save(fig, out, "chunk_size.png")


def threshold(out: Path, ev: dict) -> Path:
    a = ev["answers"]
    c = a["threshold_curve_dev"]
    xs = [p["threshold"] for p in c]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for j, (key, label) in enumerate([("span_accuracy", "answerable: correct answer given"),
                                      ("abstention_recall", "unanswerable: abstained"),
                                      ("decision_accuracy", "all: right outcome")]):
        ax.plot(xs, [p[key] for p in c], lw=2, color=SERIES[j], label=label)
    ax.axvline(a["threshold"], color=INK, lw=1.2, ls="--")
    ax.text(a["threshold"] + 0.006, 0.05, f"chosen {a['threshold']:.3f}", fontsize=8.5, color=INK)
    ax.set_xlabel("confidence threshold")
    ax.set_ylim(0, 1.05)
    ax.grid(color=GRID)
    ax.legend(frameon=False, fontsize=8.5, loc="center left")
    t = a["test"]
    ax.set_title(f"Abstention threshold chosen on the dev split (curves: dev)\nHeld-out test: "
                 f"{t['span_accuracy']:.0%} correct answers, {t['abstention_recall']:.0%} of unanswerable "
                 f"questions abstained")
    return _save(fig, out, "threshold.png")


def answer_card(out: Path, samples: list[dict]) -> Path:
    step = 1.6
    h = 0.9 + step * len(samples)
    fig, ax = plt.subplots(figsize=(12, h))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.text(0.1, h - 0.35, "Sample answers from `python -m ragqa ask` (SYNTHETIC corpus)", fontsize=12.5,
            weight="bold", color=INK)
    y = h - 0.75
    for s in samples:
        color = BAD if s["abstained"] else GOOD
        ax.add_patch(FancyBboxPatch((0.1, y - 1.4), 11.8, 1.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc="white", ec=color, lw=1.8))
        ax.text(0.3, y - 0.32, "Q: " + s["question"], fontsize=10.5, weight="bold", color=INK)
        ax.text(11.7, y - 0.32, "abstained" if s["abstained"] else "answered", fontsize=9, weight="bold",
                color=color, ha="right")
        ans = "\n".join(textwrap.wrap("A: " + s["answer_with_citations"], 125))
        ax.text(0.3, y - 0.58, ans, fontsize=9.5, color=INK, va="top", linespacing=1.4)
        top = s["retrieved"][0] if s["retrieved"] else {}
        meta = (f"confidence {s['confidence']:.3f} (threshold {s['threshold']:.3f})  ·  "
                f"top chunk: {top.get('doc', '-')} p.{','.join(map(str, top.get('pages', [])))}"
                f"{' (demoted: superseded)' if top.get('demoted') else ''}  ·  "
                f"retrieval {s['latency_ms']['retrieval']:.1f} ms")
        if s.get("gold"):
            meta += f"  ·  gold: {s['gold']}"
        ax.text(0.3, y - 1.25, meta, fontsize=8, color=MUTED)
        y -= step
    return _save(fig, out, "answer_card.png")


def render_all(reports: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    ev = json.loads((reports / "eval.json").read_text())
    paths = [architecture(out), retrieval_comparison(out, ev), recall_at_k(out, ev), threshold(out, ev)]
    if ev.get("chunk_sweep"):
        paths.append(chunk_size(out, ev))
    sp = reports / "sample_answers.json"
    if sp.exists():
        paths.append(answer_card(out, json.loads(sp.read_text())))
    return paths
