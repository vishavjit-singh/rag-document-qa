"""The same API on FastAPI (OpenAPI docs at /docs). Requires ``pip install fastapi``.

Request models, error format and the engine are shared with the Starlette app in ``api.py``, so both apps pass
the same tests (the FastAPI variant runs in CI).
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from .api import AskRequest, do_ask, do_ingest, error
from .service import RagEngine


def create_app(engine: RagEngine | None = None) -> FastAPI:
    engine = engine or RagEngine()
    app = FastAPI(title="rag-document-qa", version="1.0.0",
                  description="Retrieval-augmented Q&A over PDFs with page-level citations.")
    app.state.engine = engine

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        errs = exc.errors()
        if any(e.get("type") == "json_invalid" for e in errs):
            return error(400, "malformed_json", "request body is not valid JSON")
        return error(422, "validation_error", "request validation failed",
                     errors=[{"loc": ".".join(str(x) for x in e["loc"] if x != "body"), "msg": e["msg"]}
                             for e in errs])

    @app.get("/health")
    def health():
        return engine.health()

    @app.get("/documents")
    def documents():
        return {"documents": engine.documents()}

    @app.post("/ask")
    def ask(req: AskRequest):
        return do_ask(engine, req)

    @app.post("/ingest")
    async def ingest(request: Request):
        """JSON ``{"rebuild": false}`` re-scans the corpus folder.

        ``application/pdf`` with ``?filename=<name>.pdf`` stores the file and indexes it."""
        return await do_ingest(engine, request)

    return app


def __getattr__(name):
    if name == "app":
        return create_app()
    raise AttributeError(name)
