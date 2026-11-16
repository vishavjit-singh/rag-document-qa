"""HTTP API: POST /ask, POST /ingest, GET /documents, GET /health.

There are two apps over the same engine and pydantic models: this Starlette app, and a FastAPI app in
``api_fastapi.py`` (adds OpenAPI docs at /docs). FastAPI could not be installed where this was built, so the
Starlette app is the one tested locally; CI runs the same API tests against both. ``python -m ragqa serve``
uses FastAPI when it is installed and falls back to Starlette.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .service import RagEngine

MAX_PDF_BYTES = 20 * 1024 * 1024
SAFE_NAME = re.compile(r"^[A-Za-z0-9][\w.-]{0,120}\.pdf$")


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=500)
    k: int | None = Field(default=None, ge=1, le=20)
    mode: Literal["bm25", "dense", "hybrid"] | None = None
    answerer: Literal["extractive", "openai"] | None = None
    version_aware: bool | None = None
    mmr: bool | None = None


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rebuild: bool = False


def error(status: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, **extra}}, status_code=status)


def validation_errors(exc: ValidationError) -> list[dict]:
    return [{"loc": ".".join(str(x) for x in e["loc"]), "msg": e["msg"]} for e in exc.errors()]


def do_ask(engine: RagEngine, req: AskRequest) -> JSONResponse:
    try:
        out = engine.ask(req.question, req.k, req.mode, req.answerer, req.version_aware, req.mmr)
    except RuntimeError as exc:            # e.g. openai requested without OPENAI_API_KEY
        return error(400, "answerer_unavailable", str(exc))
    return JSONResponse(out)


async def do_ingest(engine: RagEngine, request: Request) -> JSONResponse:
    ctype = request.headers.get("content-type", "").split(";")[0].strip()
    if ctype == "application/pdf":
        name = request.query_params.get("filename", "")
        if not SAFE_NAME.match(name):
            return error(422, "bad_filename", "pass ?filename=<name>.pdf (letters, digits, '.', '-', '_')")
        body = await request.body()
        if len(body) > MAX_PDF_BYTES:
            return error(413, "too_large", f"PDF larger than {MAX_PDF_BYTES // (1024 * 1024)} MB")
        if not body.startswith(b"%PDF-"):
            return error(422, "not_a_pdf", "body does not start with %PDF-")
        engine.corpus_dir.mkdir(parents=True, exist_ok=True)
        (engine.corpus_dir / name).write_bytes(body)
        return JSONResponse({"stored": name, **engine.ingest()})
    if ctype not in ("application/json", ""):
        return error(415, "unsupported_media_type", "send application/json or application/pdf")
    raw = await request.body()
    try:
        req = IngestRequest.model_validate_json(raw or b"{}")
    except ValidationError as exc:
        if any(e["type"] == "json_invalid" for e in exc.errors()):
            return error(400, "malformed_json", "request body is not valid JSON")
        return error(422, "validation_error", "request validation failed", errors=validation_errors(exc))
    return JSONResponse(engine.ingest(rebuild=req.rebuild))


def create_app(engine: RagEngine | None = None) -> Starlette:
    engine = engine or RagEngine()

    async def health(request: Request):
        return JSONResponse(engine.health())

    async def documents(request: Request):
        return JSONResponse({"documents": engine.documents()})

    async def ask(request: Request):
        raw = await request.body()
        try:
            req = AskRequest.model_validate_json(raw)
        except ValidationError as exc:
            if any(e["type"] == "json_invalid" for e in exc.errors()):
                return error(400, "malformed_json", "request body is not valid JSON")
            return error(422, "validation_error", "request validation failed", errors=validation_errors(exc))
        return do_ask(engine, req)

    async def ingest(request: Request):
        return await do_ingest(engine, request)

    app = Starlette(routes=[Route("/health", health), Route("/documents", documents),
                            Route("/ask", ask, methods=["POST"]), Route("/ingest", ingest, methods=["POST"])])
    app.state.engine = engine
    return app


def __getattr__(name):            # `uvicorn ragqa.api:app` builds the app lazily (the index must exist)
    if name == "app":
        return create_app()
    raise AttributeError(name)
