"""A tiny ASGI client for tests: starlette's TestClient needs httpx, which is not always installed."""
from __future__ import annotations

import asyncio
import json as jsonlib
from dataclasses import dataclass


@dataclass
class Response:
    status_code: int
    content: bytes

    def json(self):
        return jsonlib.loads(self.content)


def call(app, method: str, path: str, json=None, body: bytes = b"", headers: dict | None = None,
         query: str = "") -> Response:
    headers = dict(headers or {})
    if json is not None:
        body = jsonlib.dumps(json).encode()
        headers.setdefault("content-type", "application/json")
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "path": path,
             "raw_path": path.encode(), "query_string": query.encode(), "root_path": "", "scheme": "http",
             "server": ("testserver", 80), "client": ("127.0.0.1", 50000),
             "headers": [(k.lower().encode(), v.encode()) for k, v in
                         {**headers, "content-length": str(len(body))}.items()]}

    async def run():
        inbox = [{"type": "http.request", "body": body, "more_body": False}]
        sent = []

        async def receive():
            return inbox.pop(0) if inbox else {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        await app(scope, receive, send)
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        return Response(status, b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))

    return asyncio.run(run())
