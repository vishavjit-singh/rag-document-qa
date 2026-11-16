"""``python -m ragqa generate|ingest|ask|eval|report|serve|summary``."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from . import config as config_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@click.group()
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, dir_okay=False),
              help="settings file (default: ragqa.yaml)")
@click.pass_context
def cli(ctx, config_path):
    """Retrieval-augmented question answering over PDFs, with citations."""
    ctx.obj = config_mod.load(config_path)


@cli.command()
@click.pass_obj
def generate(cfg):
    """Write the SYNTHETIC PDF corpus and the gold QA set."""
    from .synth.build import build_corpus

    stats = build_corpus(cfg["paths"]["corpus"], cfg["paths"]["gold"])
    click.echo(json.dumps({k: v for k, v in stats.items() if k != "pages_per_doc"}))


@cli.command()
@click.option("--corpus", type=click.Path(file_okay=False), default=None, help="folder of PDFs")
@click.option("--embedder", type=click.Choice(["lsa", "hashing", "sentence-transformers"]), default=None)
@click.option("--engine", type=click.Choice(["pdfplumber", "pypdf"]), default=None)
@click.option("--rebuild", is_flag=True, help="ignore caches and rebuild everything")
@click.pass_obj
def ingest(cfg, corpus, embedder, engine, rebuild):
    """Extract, chunk, de-duplicate, embed and index (incremental)."""
    from .store import ingest as run_ingest

    cfg = config_mod.with_overrides(cfg, embedding={"backend": embedder}, extract={"engine": engine})
    stats = run_ingest(Path(corpus) if corpus else cfg["paths"]["corpus"], cfg["paths"]["index"], cfg, rebuild)
    click.echo(json.dumps(stats))


@cli.command()
@click.argument("question")
@click.option("-k", type=int, default=None)
@click.option("--mode", type=click.Choice(["bm25", "dense", "hybrid"]), default=None)
@click.option("--answerer", type=click.Choice(["extractive", "openai"]), default=None)
@click.option("--json", "as_json", is_flag=True, help="print the full response")
@click.pass_obj
def ask(cfg, question, k, mode, answerer, as_json):
    """Answer a question with citations, or say it is not in the documents."""
    from .service import RagEngine

    out = RagEngine(cfg).ask(question, k=k, mode=mode, backend=answerer)
    if as_json:
        click.echo(json.dumps(out, indent=2))
        return
    click.echo(out["answer_with_citations"])
    conf = f"{out['confidence']:.3f}" if out["confidence"] is not None else "n/a"
    click.echo(f"  confidence {conf} | {out['backend']} | {out['mode']} | retrieval "
               f"{out['latency_ms']['retrieval']} ms")
    for h in out["retrieved"]:
        tag = " (ranked down: other version)" if h["demoted"] else ""
        click.echo(f"  #{h['rank']} {h['doc']} p.{','.join(map(str, h['pages']))} | {h['section']}{tag}")


@cli.command("eval")
@click.option("--no-sweep", is_flag=True, help="skip the chunk-size sweep and the extractor comparison")
@click.pass_obj
def eval_cmd(cfg, no_sweep):
    """Evaluate retrieval and answers against the gold set; calibrate the abstention threshold."""
    from .evaluate import load_gold, run
    from .service import RagEngine

    res = run(cfg, sweep=not no_sweep)
    gold = {q["id"]: q for q in load_gold(cfg["paths"]["gold"])}
    engine = RagEngine(cfg)
    samples = []
    for qid in cfg["eval"].get("sample_questions", []):
        if qid in gold:
            q = gold[qid]
            a = engine.ask(q["question"])
            a["gold"] = f"{q['doc']} p.{q['page']}, \"{q['answer']}\"" if q["answerable"] else "unanswerable"
            samples.append(a)
    (Path(cfg["paths"]["reports"]) / "sample_answers.json").write_text(json.dumps(samples, indent=2))
    _print_summary(res)


def _print_summary(res: dict):
    c, g = res["corpus"], res["gold"]
    click.echo(f"corpus: {c['documents']} PDFs, {c['pages']} pages, {c['chunks']} chunks "
               f"({c['duplicates_dropped']} duplicates dropped), embedder {c['embedder']}, index {c['index_backend']}")
    click.echo(f"gold: {g['questions']} questions ({g['answerable']} answerable, {g['unanswerable']} unanswerable), "
               f"dev {g['dev']} / test {g['test']}")
    click.echo(f"{'retrieval':32s} R@1    R@3    R@5    MRR@10 nDCG@10 ver-R@1 p50ms")
    for n, m in res["retrieval"].items():
        click.echo(f"{n:32s} {m['recall@1']:.3f}  {m['recall@3']:.3f}  {m['recall@5']:.3f}  {m['mrr@10']:.3f}  "
                   f"{m['ndcg@10']:.3f}   {m['version_questions_recall@1']:.3f}   {m['latency_ms_p50']}")
    a = res["answers"]
    click.echo(f"answers: threshold {a['threshold']} (dev); ask latency p50 {a['ask_latency_ms_p50']} ms, "
               f"p95 {a['ask_latency_ms_p95']} ms")
    for split in ("dev", "test", "all"):
        m = a[split]
        click.echo(f"  {split:4s} span acc {m['span_accuracy']:.3f} | citation acc {m['citation_accuracy']:.3f} | "
                   f"no-abstention span acc {m['span_accuracy_no_abstention']:.3f} | abstention P "
                   f"{m['abstention_precision']} R {m['abstention_recall']:.3f} | right outcome "
                   f"{m['decision_accuracy']:.3f}")
    for s in res.get("chunk_sweep", []):
        click.echo(f"  chunk size {s['size']:4d}: {s['chunks']:4d} chunks, R@1 {s['recall@1']:.3f}, "
                   f"R@5 {s['recall@5']:.3f}, span acc {s['span_accuracy_no_abstention']:.3f}")
    for e in res.get("extraction", []):
        click.echo(f"  extractor {e['engine']:10s}: {e['chunks']} chunks, R@1 {e['recall@1']:.3f}, "
                   f"R@5 {e['recall@5']:.3f}, span acc {e['span_accuracy_no_abstention']:.3f}")


@cli.command()
@click.pass_obj
def summary(cfg):
    """Print the last evaluation."""
    _print_summary(json.loads((Path(cfg["paths"]["reports"]) / "eval.json").read_text()))


@cli.command()
@click.option("--out", default=None, help="image folder (default docs/images)")
@click.pass_obj
def report(cfg, out):
    """Render the README images from reports/."""
    from .report import render_all

    for p in render_all(Path(cfg["paths"]["reports"]), Path(out) if out else cfg["paths"]["images"]):
        click.echo(f"wrote {p}")


@cli.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000, type=int)
@click.option("--framework", type=click.Choice(["auto", "fastapi", "starlette"]), default="auto")
def serve(host, port, framework):
    """Run the HTTP API (FastAPI if installed, else Starlette)."""
    import importlib.util

    import uvicorn

    use_fastapi = framework == "fastapi" or (framework == "auto" and importlib.util.find_spec("fastapi"))
    target = "ragqa.api_fastapi:app" if use_fastapi else "ragqa.api:app"
    click.echo(f"serving {target}")
    uvicorn.run(target, host=host, port=port, log_level="info")


def main():
    cli()
