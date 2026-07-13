"""Tests for BodyLimitMiddleware: rejects bodies > 10 MB with 413."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.body_limit_middleware import BodyLimitMiddleware, _10MB
from app.middleware.request_id_middleware import RequestIdMiddleware


def _make_app(max_bytes: int = _10MB) -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(BodyLimitMiddleware, max_bytes=max_bytes)

    @app.post("/upload")
    async def upload():
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


class TestBodyLimitMiddleware:
    def test_small_body_passes(self):
        client = _make_app()
        resp = client.post("/upload", content=b"x" * 100)
        assert resp.status_code == 200

    def test_exactly_at_limit_passes(self):
        client = _make_app(max_bytes=1024)
        resp = client.post("/upload", content=b"x" * 1024)
        assert resp.status_code == 200

    def test_over_limit_via_content_length_returns_413(self):
        client = _make_app(max_bytes=1024)
        data = b"x" * 1025
        resp = client.post(
            "/upload",
            content=data,
            headers={"Content-Length": str(len(data))},
        )
        assert resp.status_code == 413

    def test_413_response_envelope(self):
        client = _make_app(max_bytes=512)
        resp = client.post("/upload", content=b"x" * 513)
        assert resp.status_code == 413
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "payload_too_large"
        rid = data["error"]["requestId"]
        assert rid
        assert len(rid) == 21  # nanoid default size

    def test_413_request_id_matches_header(self):
        client = _make_app(max_bytes=512)
        resp = client.post("/upload", content=b"x" * 513)
        assert "x-request-id" in resp.headers
        assert resp.headers["x-request-id"] == resp.json()["error"]["requestId"]

    def test_413_honours_custom_request_id(self):
        client = _make_app(max_bytes=512)
        resp = client.post(
            "/upload",
            content=b"x" * 513,
            headers={"x-request-id": "custom-req-abc"},
        )
        assert resp.json()["error"]["requestId"] == "custom-req-abc"
        assert resp.headers["x-request-id"] == "custom-req-abc"
