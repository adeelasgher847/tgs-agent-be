"""
Integration tests verifying every error path returns the standard envelope:
  { "error": { "code": <snake_case>, "message": <str>, "requestId": <str> } }
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.core.exception_handlers import register_exception_handlers
from app.middleware.request_id_middleware import RequestIdMiddleware


def _make_app() -> tuple[FastAPI, TestClient]:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)

    @app.get("/ok")
    def ok():
        return {"ok": True}

    @app.get("/raise-http")
    def raise_http():
        raise HTTPException(status_code=403, detail="Forbidden")

    @app.get("/raise-unhandled")
    def raise_unhandled():
        raise RuntimeError("boom")

    @app.post("/validate")
    def validate(body: _Item):
        return body

    client = TestClient(app, raise_server_exceptions=False)
    return app, client


class _Item(BaseModel):
    name: str


def _assert_envelope(data: dict, expected_code: str | None = None):
    assert "error" in data, f"Missing 'error' key in {data}"
    err = data["error"]
    assert "code" in err
    assert "message" in err
    assert "requestId" in err
    assert err["code"] == err["code"].lower(), "error code must be snake_case"
    if expected_code:
        assert err["code"] == expected_code


class TestErrorEnvelope:
    def setup_method(self):
        _, self.client = _make_app()

    def test_http_exception_envelope(self):
        resp = self.client.get("/raise-http")
        assert resp.status_code == 403
        _assert_envelope(resp.json(), "forbidden")

    def test_unhandled_exception_envelope(self):
        resp = self.client.get("/raise-unhandled")
        assert resp.status_code == 500
        _assert_envelope(resp.json(), "internal_error")
        # Must not leak exception text.
        assert "boom" not in resp.text

    def test_validation_error_envelope(self):
        resp = self.client.post("/validate", json={})
        assert resp.status_code == 422
        _assert_envelope(resp.json(), "validation_error")

    def test_404_envelope(self):
        resp = self.client.get("/nonexistent-route")
        assert resp.status_code == 404
        data = resp.json()
        _assert_envelope(data, "not_found")
        assert data["error"]["message"] == "Not Found"

    def test_request_id_in_response_header(self):
        resp = self.client.get("/raise-http")
        assert "x-request-id" in resp.headers

    def test_request_id_propagated_to_body(self):
        resp = self.client.get("/raise-http")
        rid_header = resp.headers.get("x-request-id", "")
        rid_body = resp.json()["error"]["requestId"]
        assert rid_header == rid_body

    def test_custom_request_id_honoured(self):
        resp = self.client.get("/raise-http", headers={"x-request-id": "my-custom-id"})
        assert resp.json()["error"]["requestId"] == "my-custom-id"
