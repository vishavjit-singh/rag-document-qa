import csv
import json

from click.testing import CliRunner

from ragqa import evaluate
from ragqa.report import render_all


def test_evaluation_end_to_end(cfg):
    res = evaluate.run(cfg, sweep=False)
    r = res["retrieval"]
    best = r["Hybrid + version-aware"]
    assert best["recall@5"] >= 0.9 and best["recall@1"] > r["BM25"]["recall@1"]
    assert best["version_questions_recall@1"] > r["Hybrid (RRF)"]["version_questions_recall@1"] + 0.3
    a = res["answers"]
    assert 0 < a["threshold"] < 1 and a["test"]["span_accuracy"] >= 0.6 and a["test"]["abstention_recall"] > 0
    assert res["gold"]["dev"] + res["gold"]["test"] == res["gold"]["questions"]
    cal = json.loads((cfg["paths"]["index"] / "calibration.json").read_text())
    assert cal["threshold"] == a["threshold"]
    rows = list(csv.DictReader((cfg["paths"]["reports"] / "answers.csv").open()))
    assert len(rows) == res["gold"]["questions"] and {"dev", "test"} == {x["split"] for x in rows}


def test_cli_ask_and_report(cfg, project, tmp_path, monkeypatch):
    from ragqa.cli import cli

    # Paths in ragqa.yaml resolve against ROOT ($RAGQA_HOME or the repo), not the config file's folder,
    # so point ROOT at the temporary project instead of a data/ left behind by `make demo`.
    monkeypatch.setattr("ragqa.config.ROOT", project)
    monkeypatch.setenv("RAGQA_EMBEDDER", "lsa")
    monkeypatch.setenv("RAGQA_INDEX", "numpy")
    monkeypatch.setenv("RAGQA_ANSWERER", "extractive")
    runner = CliRunner()
    out = runner.invoke(cli, ["--config", str(project / "ragqa.yaml"), "ask", "How long is the RouteBox warranty?"])
    assert out.exit_code == 0, out.output
    assert "24 months" in out.output and "[routebox-r2-user-manual.pdf p.6]" in out.output
    if not (cfg["paths"]["reports"] / "eval.json").exists():
        evaluate.run(cfg, sweep=False)
    paths = render_all(cfg["paths"]["reports"], tmp_path)
    assert {p.name for p in paths} >= {"architecture.png", "retrieval_comparison.png", "recall_at_k.png",
                                        "threshold.png"}
    assert all(p.stat().st_size > 10_000 for p in paths)
