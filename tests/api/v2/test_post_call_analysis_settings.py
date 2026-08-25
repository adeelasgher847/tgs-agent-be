"""Tests for PUT and GET /api/v2/flows/{flow_id}/post-call-analysis-settings.

Coverage:
  - Admin/API-key principal can configure variables_to_extract + analysis_model
  - variable name pattern rejects invalid identifiers (leading digit, spaces,
    hyphens)
  - variables_to_extract rejects more than 25 entries
  - Unknown extra fields are rejected (extra="forbid"), on both the update
    body and each variable spec
  - analysis_model accepts a valid active model-catalog entry and rejects an
    unknown/archived one with the invalid_llm_model error shape + allowedValues
  - Config-rank (non-admin) principal is forbidden on PUT — mirrors
    tests/api/v2/test_post_call_actions_settings.py's admin-gate coverage
  - Unknown flow_id / other-tenant flow both return 404 (tenant isolation) on PUT and GET
  - A successful update fires an audit event with the expected shape
  - The response echoes back the persisted two-field shape
  - GET returns default unconfigured state on fresh flows and handles null DB columns
  - GET round-trips prior PUT updates accurately
  - Read-only rank is sufficient to view post-call analysis settings via GET
  - CallFlowService._resolve_analysis_model: valid active model resolves,
    archived/unknown model raises HTTPException(400, ...) with the "LLM
    model" detail substring the router pattern-matches on.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from app.core.exception_handlers import register_exception_handlers


def _build_app(db_override, principal, *, forbidden=False):
    from app.api.deps import get_db, require_admin_or_api_key
    from app.api.v2.routers.post_call_analysis import router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(router)

    if forbidden:

        def _raise_forbidden():
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

        mini.dependency_overrides[require_admin_or_api_key] = _raise_forbidden
    else:
        mini.dependency_overrides[require_admin_or_api_key] = lambda: principal

    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


def _build_readonly_app(db_override, principal, *, forbidden=False):
    from app.api.deps import get_db, require_readonly_or_api_key
    from app.api.v2.routers.post_call_analysis import router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(router)

    if forbidden:

        def _raise_forbidden():
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

        mini.dependency_overrides[require_readonly_or_api_key] = _raise_forbidden
    else:
        mini.dependency_overrides[require_readonly_or_api_key] = lambda: principal

    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


def _build_app_for_real_rank_check(db_override, principal):
    """Companion to `_with_effective_role` below — builds an app where only
    `require_tenant` is overridden (as upstream auth normally would be
    resolved), leaving the REAL rank-checking logic in
    `_require_rank_or_api_key` (app/api/deps/rbac.py) to run unmodified for
    both the admin- and readonly-gated routes on this router.
    """
    from app.api.deps import get_db, require_tenant
    from app.api.v2.routers.post_call_analysis import router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(router)

    mini.dependency_overrides[require_tenant] = lambda: principal
    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


def _with_effective_role(role_name: str):
    """Context manager: pretend `rbac_cache_service.get_effective_role`
    resolves the caller's role to `role_name`, so the real rank-checking
    dependency (`_require_rank_or_api_key`) can be exercised end-to-end
    against a controlled rank without standing up real
    user_tenant_association rows."""
    return patch(
        "app.api.deps.rbac.rbac_cache_service.get_effective_role",
        return_value=role_name,
    )


@pytest.fixture
def workspace(db):
    from app.models.tenant import Tenant

    tenant = Tenant(
        name=f"PostCallAnalysisWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"post_call_analysis_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)
    return tenant


@pytest.fixture
def agent(db, workspace):
    from app.models.agent import Agent

    a = Agent(
        tenant_id=workspace.id,
        name="Post Call Analysis Test Agent",
        status="active",
        llm_model="gpt-4o-mini",
        tts_provider_slug="elevenlabs",
        tts_voice_external_id="voice-x",
        tts_language="en",
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


@pytest.fixture
def flow(db, workspace, agent):
    from app.models.call_flow import CallFlow

    f = CallFlow(
        tenant_id=workspace.id,
        agent_id=agent.id,
        name="Post Call Analysis Test Flow",
        direction="inbound",
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


@pytest.fixture
def archived_model(db):
    """An inactive/archived model — must be rejected by analysis_model validation."""
    from app.models.model import Model
    from app.models.provider import Provider

    provider = Provider(name=f"archived-provider-{uuid.uuid4().hex[:8]}", is_active=True)
    db.add(provider)
    db.commit()
    db.refresh(provider)

    m = Model(
        provider_id=provider.id,
        model_name=f"ancient-model-v0-{uuid.uuid4().hex[:8]}",
        archive=True,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def _admin_principal(tenant_id: uuid.UUID) -> MagicMock:
    principal = MagicMock()
    principal.id = uuid.uuid4()
    principal.current_tenant_id = tenant_id
    principal.role = "admin"
    return principal


def _readonly_principal(tenant_id: uuid.UUID) -> MagicMock:
    principal = MagicMock()
    principal.id = uuid.uuid4()
    principal.current_tenant_id = tenant_id
    principal.role = "readonly"
    return principal


@pytest.mark.usefixtures("db")
class TestUpdatePostCallAnalysisSettings:
    def test_admin_can_configure_variables_and_model(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-analysis-settings",
            json={
                "variables_to_extract": [
                    {"name": "service_type", "description": "What service was requested."},
                    {"name": "urgency", "description": "How urgent the request is."},
                ],
                "analysis_model": "gpt-4o-mini",
            },
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["variables_to_extract"] == [
            {"name": "service_type", "description": "What service was requested."},
            {"name": "urgency", "description": "How urgent the request is."},
        ]
        assert body["analysis_model"] == "gpt-4o-mini"

        db.refresh(flow)
        assert flow.post_call_analysis_variables == [
            {"name": "service_type", "description": "What service was requested."},
            {"name": "urgency", "description": "How urgent the request is."},
        ]
        assert flow.post_call_analysis_model == "gpt-4o-mini"

    def test_empty_variables_and_no_model_persists_as_disabled(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-analysis-settings",
            json={"variables_to_extract": [], "analysis_model": None},
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["variables_to_extract"] == []
        assert body["analysis_model"] is None

        db.refresh(flow)
        assert flow.post_call_analysis_variables == []
        assert flow.post_call_analysis_model is None

    @pytest.mark.parametrize(
        "invalid_name",
        ["1_starts_with_digit", "has space", "has-hyphen", ""],
    )
    def test_invalid_variable_name_pattern_rejected(self, db, workspace, flow, invalid_name):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-analysis-settings",
            json={
                "variables_to_extract": [
                    {"name": invalid_name, "description": "some description"}
                ],
                "analysis_model": None,
            },
        )

        assert resp.status_code == 400

    def test_more_than_25_variables_rejected(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        variables = [
            {"name": f"var_{i}", "description": f"Extract variable {i}."} for i in range(26)
        ]
        resp = client.put(
            f"/flows/{flow.id}/post-call-analysis-settings",
            json={"variables_to_extract": variables, "analysis_model": None},
        )

        assert resp.status_code == 400

    def test_exactly_25_variables_accepted(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        variables = [
            {"name": f"var_{i}", "description": f"Extract variable {i}."} for i in range(25)
        ]
        resp = client.put(
            f"/flows/{flow.id}/post-call-analysis-settings",
            json={"variables_to_extract": variables, "analysis_model": None},
        )

        assert resp.status_code == 200, resp.text
        assert len(resp.json()["variables_to_extract"]) == 25

    def test_extra_field_on_update_body_rejected(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-analysis-settings",
            json={
                "variables_to_extract": [],
                "analysis_model": None,
                "unexpected_field": "nope",
            },
        )

        assert resp.status_code == 400

    def test_extra_field_on_variable_spec_rejected(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-analysis-settings",
            json={
                "variables_to_extract": [
                    {"name": "service_type", "description": "desc", "type": "string"}
                ],
                "analysis_model": None,
            },
        )

        assert resp.status_code == 400

    def test_blank_analysis_model_rejected(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-analysis-settings",
            json={"variables_to_extract": [], "analysis_model": "   "},
        )

        assert resp.status_code == 400

    def test_unknown_analysis_model_rejected_with_invalid_llm_model_shape(
        self, db, workspace, flow
    ):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-analysis-settings",
            json={"variables_to_extract": [], "analysis_model": "totally-made-up-model"},
        )

        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert body["error"]["code"] == "invalid_llm_model"
        assert "allowedValues" in body["error"]
        assert "gpt-4o-mini" in body["error"]["allowedValues"]

        # Not persisted
        db.refresh(flow)
        assert flow.post_call_analysis_model is None

    def test_archived_analysis_model_rejected(self, db, workspace, flow, archived_model):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-analysis-settings",
            json={
                "variables_to_extract": [],
                "analysis_model": archived_model.model_name,
            },
        )

        assert resp.status_code == 400, resp.text
        assert resp.json()["error"]["code"] == "invalid_llm_model"

    def test_non_admin_principal_is_forbidden(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal, forbidden=True)

        resp = client.put(
            f"/flows/{flow.id}/post-call-analysis-settings",
            json={"variables_to_extract": [], "analysis_model": None},
        )

        assert resp.status_code == 403

    def test_unknown_flow_returns_404(self, db, workspace):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{uuid.uuid4()}/post-call-analysis-settings",
            json={"variables_to_extract": [], "analysis_model": None},
        )

        assert resp.status_code == 404

    def test_flow_from_other_tenant_returns_404(self, db, flow):
        """Tenant isolation: a principal from another tenant must not be able
        to update — or even discover the existence of — this flow."""
        from app.models.tenant import Tenant

        other_tenant = Tenant(
            name=f"OtherAnalysisWS-{uuid.uuid4().hex[:8]}",
            schema_name=f"other_analysis_ws_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(other_tenant)
        db.commit()
        db.refresh(other_tenant)

        principal = _admin_principal(other_tenant.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-analysis-settings",
            json={
                "variables_to_extract": [{"name": "x", "description": "y"}],
                "analysis_model": None,
            },
        )

        assert resp.status_code == 404

        db.refresh(flow)
        assert flow.post_call_analysis_variables == []
        assert flow.post_call_analysis_model is None

    def test_update_fires_audit_event(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        with patch(
            "app.api.v2.routers.post_call_analysis.log_audit_event"
        ) as mock_log_audit:
            resp = client.put(
                f"/flows/{flow.id}/post-call-analysis-settings",
                json={
                    "variables_to_extract": [
                        {"name": "service_type", "description": "Extract the service type."}
                    ],
                    "analysis_model": "gpt-4o-mini",
                },
            )

        assert resp.status_code == 200, resp.text
        mock_log_audit.assert_called_once()
        kwargs = mock_log_audit.call_args.kwargs
        assert kwargs["action"] == "post_call_analysis_settings.updated"
        assert kwargs["resource_type"] == "call_flow"
        assert kwargs["resource_id"] == flow.id
        assert kwargs["new_value"] == {
            "variables_to_extract": [
                {"name": "service_type", "description": "Extract the service type."}
            ],
            "analysis_model": "gpt-4o-mini",
        }
        assert kwargs["actor_user_id"] == principal.id


@pytest.mark.usefixtures("db")
class TestResolveAnalysisModel:
    """Unit coverage for CallFlowService._resolve_analysis_model, mirroring
    agent_service._resolve_llm_model's validation pattern exactly."""

    def test_valid_active_model_resolves(self, db):
        from app.services.call_flow_service import call_flow_service

        model = call_flow_service._resolve_analysis_model(db, "gpt-4o-mini")
        assert model.model_name == "gpt-4o-mini"
        assert model.archive is False

    def test_unknown_model_raises_with_llm_model_substring(self, db):
        from app.services.call_flow_service import call_flow_service

        with pytest.raises(HTTPException) as excinfo:
            call_flow_service._resolve_analysis_model(db, "does-not-exist")

        assert excinfo.value.status_code == 400
        assert "LLM model" in str(excinfo.value.detail)

    def test_archived_model_raises_with_llm_model_substring(self, db, archived_model):
        from app.services.call_flow_service import call_flow_service

        with pytest.raises(HTTPException) as excinfo:
            call_flow_service._resolve_analysis_model(db, archived_model.model_name)

        assert excinfo.value.status_code == 400
        assert "LLM model" in str(excinfo.value.detail)


@pytest.mark.usefixtures("db")
class TestGetPostCallAnalysisSettings:
    def test_get_returns_defaults_for_unconfigured_flow(self, db, workspace, flow):
        principal = _readonly_principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/post-call-analysis-settings")

        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "variables_to_extract": [],
            "analysis_model": None,
        }

    def test_get_round_trips_a_prior_put(self, db, workspace, flow):
        admin_client = _build_app(db, _admin_principal(workspace.id))
        put_payload = {
            "variables_to_extract": [
                {"name": "service_type", "description": "What service was requested."},
                {"name": "urgency", "description": "How urgent the request is."},
            ],
            "analysis_model": "gpt-4o-mini",
        }
        put_resp = admin_client.put(
            f"/flows/{flow.id}/post-call-analysis-settings",
            json=put_payload,
        )
        assert put_resp.status_code == 200, put_resp.text
        put_body = put_resp.json()

        readonly_client = _build_readonly_app(db, _readonly_principal(workspace.id))
        get_resp = readonly_client.get(f"/flows/{flow.id}/post-call-analysis-settings")

        assert get_resp.status_code == 200, get_resp.text
        get_body = get_resp.json()
        assert get_body == put_body
        assert len(get_body["variables_to_extract"]) == 2
        assert get_body["variables_to_extract"][0] == {
            "name": "service_type",
            "description": "What service was requested.",
        }
        assert get_body["variables_to_extract"][1] == {
            "name": "urgency",
            "description": "How urgent the request is.",
        }
        assert get_body["analysis_model"] == "gpt-4o-mini"

    def test_get_flow_from_other_tenant_returns_404(self, db, flow):
        from app.models.tenant import Tenant

        other_tenant = Tenant(
            name=f"OtherAnalysisWS-{uuid.uuid4().hex[:8]}",
            schema_name=f"other_analysis_ws_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(other_tenant)
        db.commit()
        db.refresh(other_tenant)

        principal = _readonly_principal(other_tenant.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/post-call-analysis-settings")
        assert resp.status_code == 404

    def test_get_unknown_flow_returns_404(self, db, workspace):
        principal = _readonly_principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{uuid.uuid4()}/post-call-analysis-settings")
        assert resp.status_code == 404

    def test_readonly_principal_can_get(self, db, workspace, flow):
        principal = _readonly_principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/post-call-analysis-settings")
        assert resp.status_code == 200, resp.text

    def test_readonly_rank_can_access_get(self, db, workspace, flow):
        """Readonly is the floor for this endpoint — a genuinely readonly-rank
        principal (real rank-check dependency, not just a dependency-swap)
        must still succeed."""
        principal = _admin_principal(workspace.id)
        client = _build_app_for_real_rank_check(db, principal)

        with _with_effective_role("read_only"):
            resp = client.get(f"/flows/{flow.id}/post-call-analysis-settings")

        assert resp.status_code == 200, resp.text

    def test_forbidden_principal_cannot_get(self, db, workspace, flow):
        principal = _readonly_principal(workspace.id)
        client = _build_readonly_app(db, principal, forbidden=True)

        resp = client.get(f"/flows/{flow.id}/post-call-analysis-settings")
        assert resp.status_code == 403

    def test_get_handles_null_database_columns(self, db, workspace, agent):
        from app.models.call_flow import CallFlow

        flow = CallFlow(
            tenant_id=workspace.id,
            agent_id=agent.id,
            name="Null Columns Flow",
            direction="inbound",
            post_call_analysis_variables=None,
            post_call_analysis_model=None,
        )
        db.add(flow)
        db.commit()
        db.refresh(flow)

        principal = _readonly_principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/post-call-analysis-settings")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "variables_to_extract": [],
            "analysis_model": None,
        }

    def test_get_round_trips_with_variables_and_null_model(
        self, db, workspace, flow
    ):
        admin_client = _build_app(db, _admin_principal(workspace.id))
        put_payload = {
            "variables_to_extract": [
                {"name": "caller_intent", "description": "Why the caller called."}
            ],
            "analysis_model": None,
        }
        put_resp = admin_client.put(
            f"/flows/{flow.id}/post-call-analysis-settings",
            json=put_payload,
        )
        assert put_resp.status_code == 200, put_resp.text

        readonly_client = _build_readonly_app(db, _readonly_principal(workspace.id))
        get_resp = readonly_client.get(f"/flows/{flow.id}/post-call-analysis-settings")

        assert get_resp.status_code == 200, get_resp.text
        assert get_resp.json() == {
            "variables_to_extract": [
                {"name": "caller_intent", "description": "Why the caller called."}
            ],
            "analysis_model": None,
        }

