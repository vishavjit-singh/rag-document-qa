import importlib.util

import pytest

from ragqa.api import create_app as starlette_app
from ragqa.service import RagEngine
from tests.asgi import call

FRAMEWORKS = ["starlette", pytest.param("fastapi", marks=pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None, reason="fastapi not installed (runs in CI)"))]


@pytest.fixture(params=FRAMEWORKS)
def app(request, fresh):
    engine = RagEngine(fresh)
    if request.param == "fastapi":
        from ragqa.api_fastapi import create_app

        return create_app(engine)
    return starlette_app(engine)


def test_health_and_documents(app):
    h = call(app, "GET", "/health").json()
    assert h["status"] == "ok" and h["documents"] == 9 and h["embedder"] == "lsa"
    docs = call(app, "GET", "/documents").json()["documents"]
    old = next(d for d in docs if d["file"] == "travel-expense-policy-2023.pdf")
    assert old["status"] == "Superseded" and old["version"] == "2023.2" and old["pages"] == 6


def test_ask_returns_a_cited_answer(app):
    r = call(app, "POST", "/ask", json={"question": "What is the forklift speed limit indoors?"})
    assert r.status_code == 200
    body = r.json()
    assert "8 km/h" in body["answer"] and body["citations"][0]["doc"] == "warehouse-safety-procedures.pdf"
    assert body["answer_with_citations"].endswith("[warehouse-safety-procedures.pdf p.4]")
    assert body["retrieved"][0]["rank"] == 1 and "text" not in body["retrieved"][0]
    bm = call(app, "POST", "/ask", json={"question": "speed limit for forklifts", "mode": "bm25", "k": 2}).json()
    assert bm["mode"] == "bm25" and len(bm["retrieved"]) == 2


@pytest.mark.parametrize("body, field", [
    ({"question": "hi"}, "question"), ({"question": "valid question", "k": 0}, "k"),
    ({"question": "valid question", "mode": "sparse"}, "mode"), ({"question": "valid question", "colour": 1}, "colour"),
    ({}, "question")])
def test_ask_validation(app, body, field):
    r = call(app, "POST", "/ask", json=body)
    assert r.status_code == 422 and any(e["loc"].startswith(field) for e in r.json()["error"]["errors"])


def test_malformed_json_and_missing_openai_key(app, monkeypatch):
    r = call(app, "POST", "/ask", body=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "malformed_json"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = call(app, "POST", "/ask", json={"question": "forklift speed?", "answerer": "openai"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "answerer_unavailable"


def test_ingest_rescan_and_pdf_upload(app, fresh):
    assert call(app, "POST", "/ingest", json={}).json()["up_to_date"] is True
    pdf = (fresh["paths"]["corpus"] / "routebox-r2-user-manual.pdf").read_bytes()
    bad = call(app, "POST", "/ingest", body=pdf, headers={"content-type": "application/pdf"}, query="filename=../x.pdf")
    assert bad.status_code == 422
    notpdf = call(app, "POST", "/ingest", body=b"hello", headers={"content-type": "application/pdf"},
                  query="filename=a.pdf")
    assert notpdf.status_code == 422 and notpdf.json()["error"]["code"] == "not_a_pdf"
    ok = call(app, "POST", "/ingest", body=pdf, headers={"content-type": "application/pdf"},
              query="filename=routebox-copy.pdf").json()
    assert ok["stored"] == "routebox-copy.pdf" and ok["added"] == ["routebox-copy.pdf"]
    assert ok["parsed"] == 0 and ok["duplicates_dropped"] >= 5        # same bytes: cached extraction, deduped
    assert call(app, "GET", "/health").json()["documents"] == 10
    assert call(app, "POST", "/ingest", body=b"x", headers={"content-type": "text/plain"}).status_code == 415
