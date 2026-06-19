"""Unit tests for PublicSdkCorsMiddleware — dynamic CORS for the one public
Web SDK route. The static global CORSMiddleware can't allow per-workspace
allowed_domains, so this middleware reflects Origin only on that one path;
real authorization still happens in app/routers/sdk.py.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.public_sdk_cors_middleware import PublicSdkCorsMiddleware


def _app() -> FastAPI:
    mini = FastAPI()
    mini.add_middleware(PublicSdkCorsMiddleware)

    @mini.post("/api/v1/sdk/public-call-token")
    def public_sdk_token():
        return {"ok": True}

    @mini.get("/api/v1/other")
    def other():
        return {"ok": True}

    return mini


def test_preflight_reflects_arbitrary_origin():
    client = TestClient(_app())
    resp = client.options(
        "/api/v1/sdk/public-call-token",
        headers={
            "Origin": "https://totally-unlisted.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "https://totally-unlisted.example.com"


def test_actual_response_reflects_origin():
    client = TestClient(_app())
    resp = client.post(
        "/api/v1/sdk/public-call-token",
        headers={"Origin": "https://embed.example.com"},
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "https://embed.example.com"


def test_other_paths_untouched():
    client = TestClient(_app())
    resp = client.get("/api/v1/other", headers={"Origin": "https://embed.example.com"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


def test_no_origin_header_no_cors_header_added():
    client = TestClient(_app())
    resp = client.post("/api/v1/sdk/public-call-token")
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers
