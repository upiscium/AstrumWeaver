"""Real loopback transport proof that protected native bearer requests bypass ambient proxies."""
from __future__ import annotations

import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import httpx
import pytest

from astrumweaver.runtime.providers.llama_cpp import HttpLlamaCppApi


@contextmanager
def _server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.asyncio
async def test_authenticated_native_client_ignores_hostile_proxy_environment(monkeypatch):
    """The child auth token and data must never traverse an inherited proxy."""
    incoming: list[tuple[str, str | None]] = []
    diverted: list[tuple[str, str | None]] = []

    class NativeHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def _respond(self):
            incoming.append((self.path, self.headers.get("Authorization")))
            if self.path == "/v1/models":
                content = {"data": [{"id": "private-model"}]}
            elif self.path == "/v1/embeddings":
                content = {"object": "list", "data": [{"embedding": [0.1, 0.2]}]}
            elif self.path == "/v1/systemone":
                content = {"answers": {"decision": {"type": "choice", "probabilities": {"0000": 1.0}}}}
            else:
                content = {"status": "ok"}
            raw = json.dumps(content).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        do_GET = do_POST = _respond

    class ProxyHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def _respond(self):
            diverted.append((self.path, self.headers.get("Authorization")))
            raw = b'{"error":"untrusted proxy intercepted a native request"}'
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        do_GET = do_POST = _respond

    with _server(NativeHandler) as native_port, _server(ProxyHandler) as proxy_port:
        proxy = f"http://127.0.0.1:{proxy_port}"
        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(key, proxy)
        for key in ("NO_PROXY", "no_proxy"):
            monkeypatch.setenv(key, "")

        api = HttpLlamaCppApi(
            f"http://127.0.0.1:{native_port}", api_key="private-native-token",
            timeout_seconds=5,
        )
        try:
            assert await api.health()
            assert await api.models() == ("private-model",)
            assert (await api.embeddings({"input": ["synthetic note"]}))["data"]
            assert (await api.system_one({"state": "synthetic"}))["answers"]
        finally:
            await api.close()

    assert len(incoming) == 4
    assert all(token == "Bearer private-native-token" for _, token in incoming)
    assert diverted == [], "the proxy must never receive the bearer or inference request"


@pytest.mark.asyncio
async def test_authenticated_native_client_disables_proxy_trust_at_constructor(monkeypatch):
    """Supplement the network test by pinning the private HTTPX trust boundary."""
    seen: list[dict] = []
    original = httpx.AsyncClient

    def capture(*args, **kwargs):
        seen.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", capture)
    api = HttpLlamaCppApi("http://127.0.0.1:12345", api_key="synthetic-token")
    try:
        assert seen[-1].get("trust_env") is False
        assert seen[-1].get("follow_redirects") is False
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_unauthenticated_legacy_native_client_keeps_proxy_compatibility(monkeypatch):
    seen: list[dict] = []
    original = httpx.AsyncClient

    def capture(*args, **kwargs):
        seen.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", capture)
    api = HttpLlamaCppApi("http://127.0.0.1:12345")
    try:
        assert seen[-1].get("trust_env") is True
        assert seen[-1].get("follow_redirects") is False
    finally:
        await api.close()
