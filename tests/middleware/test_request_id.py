"""Tests for RequestIdMiddleware: nanoid generation and header propagation."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.middleware.request_id_middleware import RequestIdMiddleware, _NANOID_SIZE


def _make_app() -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/ping")
    def ping(request: Request):
        return {"requestId": request.state.request_id}

    return TestClient(app)


class TestRequestIdMiddleware:
    def setup_method(self):
        self.client = _make_app()

    def test_request_id_set_on_state(self):
        resp = self.client.get("/ping")
        assert resp.status_code == 200
        rid = resp.json()["requestId"]
        assert rid  # non-empty

    def test_request_id_in_response_header(self):
        resp = self.client.get("/ping")
        assert "x-request-id" in resp.headers

    def test_state_matches_header(self):
        resp = self.client.get("/ping")
        assert resp.json()["requestId"] == resp.headers["x-request-id"]

    def test_upstream_id_honoured(self):
        resp = self.client.get("/ping", headers={"x-request-id": "upstream-123"})
        assert resp.json()["requestId"] == "upstream-123"
        assert resp.headers["x-request-id"] == "upstream-123"

    def test_generated_id_is_nanoid_length(self):
        resp = self.client.get("/ping")
        rid = resp.json()["requestId"]
        assert len(rid) == _NANOID_SIZE
