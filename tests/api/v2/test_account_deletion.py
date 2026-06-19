"""
Tests for the GDPR right-to-erasure endpoint.

Coverage:
  - POST /workspace/account/delete: wrong confirmation phrase -> 400, service untouched
  - POST /workspace/account/delete: case-sensitive mismatch -> 400
  - POST /workspace/account/delete: missing body field -> 400
  - POST /workspace/account/delete: exact phrase -> 204, audits then calls the wipe service
  - Admin RBAC required

POST (not DELETE-with-body) because proxies/load balancers aren't guaranteed
to forward a body on DELETE — see app/api/v2/routers/workspace.py docstring.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from app.api.deps import get_db, require_admin
from app.core.exception_handlers import register_exception_handlers

WORKSPACE_ID = uuid.uuid4()
USER_ID = uuid.uuid4()

_CORRECT_PHRASE = "DELETE MY ACCOUNT"


def _make_admin_user() -> MagicMock:
    user = MagicMock()
    user.id = USER_ID
    user.current_tenant_id = WORKSPACE_ID
    return user


def _build_app(db_override, *, admin_override=None) -> TestClient:
    from app.api.v2.routers.workspace import router as workspace_router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(workspace_router)

    mini.dependency_overrides[require_admin] = admin_override or (lambda: _make_admin_user())
    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


class TestDeleteAccount:
    def test_wrong_phrase_returns_400_and_does_not_delete(self):
        db = MagicMock()

        with patch("app.api.v2.routers.workspace.delete_workspace_account") as mock_delete:
            client = _build_app(db)
            resp = client.post(
                "/workspace/account/delete", json={"confirmation": "delete my account please"}
            )

        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        mock_delete.assert_not_called()

    def test_case_mismatch_returns_400(self):
        db = MagicMock()

        with patch("app.api.v2.routers.workspace.delete_workspace_account") as mock_delete:
            client = _build_app(db)
            resp = client.post(
                "/workspace/account/delete", json={"confirmation": "delete my account"}
            )

        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        mock_delete.assert_not_called()

    def test_missing_confirmation_field_returns_400(self):
        db = MagicMock()
        client = _build_app(db)

        resp = client.post("/workspace/account/delete", json={})

        assert resp.status_code == status.HTTP_400_BAD_REQUEST

    def test_correct_phrase_returns_204_and_wipes(self):
        db = MagicMock()

        with (
            patch("app.api.v2.routers.workspace.delete_workspace_account") as mock_delete,
            patch("app.api.v2.routers.workspace.log_audit_event") as mock_audit,
        ):
            client = _build_app(db)
            resp = client.post(
                "/workspace/account/delete", json={"confirmation": _CORRECT_PHRASE}
            )

        assert resp.status_code == status.HTTP_204_NO_CONTENT, resp.text
        assert resp.content == b""
        mock_delete.assert_called_once_with(db, WORKSPACE_ID)
        mock_audit.assert_called_once()
        assert mock_audit.call_args.kwargs["action"] == "workspace.account_deleted"

    def test_audit_logged_before_wipe(self):
        """The deletion event must be recorded before the wipe runs (order matters:
        the wipe's auditlog UPDATE sweeps up this very row too)."""
        db = MagicMock()
        call_order = []

        with (
            patch(
                "app.api.v2.routers.workspace.delete_workspace_account",
                side_effect=lambda *a, **k: call_order.append("delete"),
            ),
            patch(
                "app.api.v2.routers.workspace.log_audit_event",
                side_effect=lambda *a, **k: call_order.append("audit"),
            ),
        ):
            client = _build_app(db)
            client.post("/workspace/account/delete", json={"confirmation": _CORRECT_PHRASE})

        assert call_order == ["audit", "delete"]

    def test_requires_admin(self):
        db = MagicMock()

        def _forbidden():
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")

        client = _build_app(db, admin_override=_forbidden)
        resp = client.post("/workspace/account/delete", json={"confirmation": _CORRECT_PHRASE})

        assert resp.status_code == status.HTTP_403_FORBIDDEN
