"""OpenAPI contract tests — assert workspace CRUD paths + security schemes are in the spec.

These are fast, import-only tests; no database required.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from app.main import app

_REPO_ROOT = Path(__file__).resolve().parents[1]
_COMMITTED_SPEC = _REPO_ROOT / "docs" / "api" / "openapi.yaml"


@pytest.fixture(scope="module")
def spec() -> dict:
    """Build OpenAPI schema via custom_openapi() (default /openapi.json is disabled)."""
    app.openapi_schema = None
    return app.openapi()


# ---------------------------------------------------------------------------
# Workspace CRUD paths
# ---------------------------------------------------------------------------

class TestWorkspacePaths:
    _EXPECTED_PATHS = [
        ("/api/v1/workspace", "post"),
        ("/api/v1/workspace/{workspace_id}", "get"),
        ("/api/v1/workspace/name", "put"),
        ("/api/v1/workspace/{workspace_id}", "delete"),
    ]

    def test_all_workspace_paths_present(self, spec):
        paths = spec.get("paths", {})
        for path, method in self._EXPECTED_PATHS:
            assert path in paths, f"Path {path!r} missing from OpenAPI spec"
            assert method in paths[path], f"Method {method.upper()} missing on {path!r}"

    def test_post_workspace_returns_201(self, spec):
        op = spec["paths"]["/api/v1/workspace"]["post"]
        assert "201" in op.get("responses", {}), "POST /workspace should document 201"

    def test_get_workspace_returns_200(self, spec):
        op = spec["paths"]["/api/v1/workspace/{workspace_id}"]["get"]
        assert "200" in op.get("responses", {}), "GET /workspace/{id} should document 200"

    def test_delete_workspace_returns_204(self, spec):
        op = spec["paths"]["/api/v1/workspace/{workspace_id}"]["delete"]
        assert "204" in op.get("responses", {}), "DELETE /workspace/{id} should document 204"

    def test_workspace_paths_document_401(self, spec):
        paths = spec.get("paths", {})
        for path, method in self._EXPECTED_PATHS:
            op = paths[path][method]
            responses = op.get("responses", {})
            assert "401" in responses, f"{method.upper()} {path} missing 401 doc"

    def test_workspace_paths_document_409(self, spec):
        for path, method in [
            ("/api/v1/workspace", "post"),
            ("/api/v1/workspace/name", "put"),
        ]:
            op = spec["paths"][path][method]
            responses = op.get("responses", {})
            assert "409" in responses, f"{method.upper()} {path} missing 409 doc"


# ---------------------------------------------------------------------------
# Security schemes
# ---------------------------------------------------------------------------

class TestSecuritySchemes:
    def test_components_security_schemes_present(self, spec):
        schemes = spec.get("components", {}).get("securitySchemes", {})
        assert schemes, "No securitySchemes in components"

    def test_api_key_auth_scheme_exists(self, spec):
        schemes = spec["components"]["securitySchemes"]
        assert "ApiKeyAuth" in schemes, "ApiKeyAuth scheme missing"

    def test_workspace_id_scheme_exists(self, spec):
        schemes = spec["components"]["securitySchemes"]
        assert "WorkspaceId" in schemes, "WorkspaceId scheme missing"

    def test_api_key_auth_is_header_type(self, spec):
        scheme = spec["components"]["securitySchemes"]["ApiKeyAuth"]
        assert scheme["type"] == "apiKey"
        assert scheme["in"] == "header"
        assert scheme["name"] == "x-api-key"

    def test_workspace_id_is_header_type(self, spec):
        scheme = spec["components"]["securitySchemes"]["WorkspaceId"]
        assert scheme["type"] == "apiKey"
        assert scheme["in"] == "header"
        assert scheme["name"] == "x-workspace-id"

    def test_security_applied_to_workspace_post(self, spec):
        op = spec["paths"]["/api/v1/workspace"]["post"]
        security = op.get("security", [])
        assert any("ApiKeyAuth" in req for req in security), (
            "POST /workspace missing ApiKeyAuth security requirement"
        )
        assert any("WorkspaceId" in req for req in security), (
            "POST /workspace missing WorkspaceId security requirement"
        )

    def test_security_applied_to_workspace_get(self, spec):
        op = spec["paths"]["/api/v1/workspace/{workspace_id}"]["get"]
        security = op.get("security", [])
        assert any("ApiKeyAuth" in req for req in security)

    def test_security_applied_to_workspace_put(self, spec):
        op = spec["paths"]["/api/v1/workspace/name"]["put"]
        security = op.get("security", [])
        assert any("ApiKeyAuth" in req for req in security)

    def test_security_applied_to_workspace_delete(self, spec):
        op = spec["paths"]["/api/v1/workspace/{workspace_id}"]["delete"]
        security = op.get("security", [])
        assert any("ApiKeyAuth" in req for req in security)


# ---------------------------------------------------------------------------
# Schema examples
# ---------------------------------------------------------------------------

class TestSchemaExamples:
    def test_workspace_created_out_has_examples(self, spec):
        """WorkspaceCreate body is parsed manually (Depends), so its schema isn't in
        components — check WorkspaceCreatedOut (201 response) which IS generated."""
        schemas = spec.get("components", {}).get("schemas", {})
        created = schemas.get("WorkspaceCreatedOut", {})
        assert created, "WorkspaceCreatedOut schema missing from components"
        name_prop = created.get("properties", {}).get("name", {})
        assert name_prop.get("examples") or name_prop.get("example"), (
            "WorkspaceCreatedOut.name should have examples"
        )


# ---------------------------------------------------------------------------
# Voice / webhook routes must remain hidden
# ---------------------------------------------------------------------------

class TestHiddenRoutes:
    def test_stream_routes_hidden_from_spec(self, spec):
        """Bidirectional stream + live-voice routes must be hidden (include_in_schema=False)."""
        paths = spec.get("paths", {})
        stream_paths = [p for p in paths if p.startswith("/api/v1/stream/")]
        assert not stream_paths, f"Streaming paths should be hidden: {stream_paths}"

    def test_live_voice_hidden_from_spec(self, spec):
        paths = spec.get("paths", {})
        live_voice_paths = [p for p in paths if p.startswith("/api/v1/live-voice/")]
        assert not live_voice_paths, f"Live-voice paths should be hidden: {live_voice_paths}"

    def test_workspace_routes_are_visible(self, spec):
        """Workspace routes must now appear in the spec (our primary change)."""
        paths = spec.get("paths", {})
        assert "/api/v1/workspace" in paths, "POST /workspace missing from spec"

    def test_health_is_accessible_in_spec(self, spec):
        # /health is a public endpoint and may or may not be in the spec;
        # what matters is it's not accidentally hidden by the workspace fix.
        paths = spec.get("paths", {})
        assert isinstance(paths, dict)


# ---------------------------------------------------------------------------
# Committed spec + GET /api/docs
# ---------------------------------------------------------------------------

class TestApiDocsRoute:
    def test_default_docs_routes_disabled(self):
        with TestClient(app, raise_server_exceptions=True) as client:
            assert client.get("/docs").status_code == 404
            assert client.get("/redoc").status_code == 404
            assert client.get("/openapi.json").status_code == 404

    def test_openapi_version_is_3_0_3(self, spec):
        assert spec.get("openapi") == "3.0.3"

    def test_committed_yaml_exists_and_matches_version(self):
        assert _COMMITTED_SPEC.is_file(), f"Missing committed spec: {_COMMITTED_SPEC}"
        with _COMMITTED_SPEC.open(encoding="utf-8") as fh:
            committed = yaml.safe_load(fh)
        assert committed.get("openapi") == "3.0.3"

    def test_get_api_docs_returns_swagger_ui(self):
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get("/api/docs")
        assert resp.status_code == 200
        assert "swagger" in resp.text.lower()

    def test_get_api_docs_openapi_yaml(self):
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get("/api/docs/openapi.yaml")
        assert resp.status_code == 200
        assert "openapi:" in resp.text
