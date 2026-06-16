"""
Tests for v2 Active Call Insights endpoints.

Coverage:
  1. GET /active-calls returns list of active call sessions for the workspace
  2. GET /active-calls returns empty list when no active calls
  3. GET /active-calls/{room_name}/insights — happy path: mocked Gemini returns valid JSON
  4. GET /active-calls/{room_name}/insights — Gemini error → fallback neutral response
  5. GET /active-calls/{room_name}/insights — no Redis transcript → fallback neutral response
  6. GET /active-calls/{room_name}/insights — room belongs to different workspace → 401
  7. GET /active-calls/{room_name}/insights — rate limit exceeded → 429
  8. GET /active-calls/{room_name}/insights — malformed Gemini JSON → fallback neutral response
  9. GET /active-calls/{room_name}/insights — sentiment clamping (out-of-range score)
 10. GET /active-calls — agents without a name return "Unknown Agent"
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_db, get_workspace
from app.core.exception_handlers import register_exception_handlers
from app.core.workspace import Workspace


# ── Constants ─────────────────────────────────────────────────────────────────

WORKSPACE_ID = uuid.uuid4()
OTHER_WORKSPACE_ID = uuid.uuid4()
CALL_SESSION_ID = uuid.uuid4()
AGENT_ID = uuid.uuid4()
ROOM_NAME = f"room_{CALL_SESSION_ID}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_workspace(ws_id: uuid.UUID = WORKSPACE_ID) -> Workspace:
    ws = MagicMock(spec=Workspace)
    ws.id = ws_id
    return ws


def _make_session(
    *,
    session_id: uuid.UUID = CALL_SESSION_ID,
    tenant_id: uuid.UUID = WORKSPACE_ID,
    status: str = "active",
    from_number: str = "+15550001001",
    to_number: str = "+15550002002",
    start_time: datetime | None = None,
) -> MagicMock:
    from app.models.call_session import CallSession

    cs = MagicMock(spec=CallSession)
    cs.id = session_id
    cs.tenant_id = tenant_id
    cs.agent_id = AGENT_ID
    cs.status = status
    cs.from_number = from_number
    cs.to_number = to_number
    cs.start_time = start_time or datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return cs


def _make_agent(name: str = "Sales Bot") -> MagicMock:
    from app.models.agent import Agent

    ag = MagicMock(spec=Agent)
    ag.id = AGENT_ID
    ag.name = name
    return ag


def _build_app(
    *,
    db_sessions: list | None = None,
    ws_id: uuid.UUID = WORKSPACE_ID,
) -> TestClient:
    """
    Minimal FastAPI app with only the active-calls router.
    DB is replaced by a mock that returns db_sessions from execute().
    """
    from app.api.v2.routers import active_calls as ac_module

    ws = _mock_workspace(ws_id)
    sessions = db_sessions or []

    # Build mock db that handles both execute() (for list) and get() (for insights).
    mock_db = MagicMock()

    # execute() returns rows as (CallSession, agent_name) tuples
    mock_rows = [(s, _make_agent().name) for s in sessions]
    mock_result = MagicMock()
    mock_result.all.return_value = mock_rows
    mock_db.execute.return_value = mock_result

    # db.get() returns the session matching the ID (used by insights endpoint)
    def _db_get(model, pk):
        for s in sessions:
            if str(s.id) == str(pk):
                return s
        return None

    mock_db.get.side_effect = _db_get

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(ac_module.router)

    mini.dependency_overrides[get_workspace] = lambda: ws
    mini.dependency_overrides[get_db] = lambda: mock_db

    return TestClient(mini, raise_server_exceptions=False)


# ── GET /active-calls ─────────────────────────────────────────────────────────

class TestListActiveCalls:
    def test_returns_active_calls(self):
        session = _make_session()
        client = _build_app(db_sessions=[session])

        resp = client.get("/active-calls")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        item = data[0]
        assert item["room_name"] == ROOM_NAME
        assert item["call_id"] == str(CALL_SESSION_ID)
        assert item["agent_name"] == "Sales Bot"
        assert item["from_number"] == "+15550001001"
        assert item["to_number"] == "+15550002002"
        assert item["duration_seconds"] >= 0

    def test_returns_empty_list_when_no_active_calls(self):
        client = _build_app(db_sessions=[])
        resp = client.get("/active-calls")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_agent_name_fallback(self):
        """When agent join returns null name, fallback to 'Unknown Agent'."""
        session = _make_session()
        ws = _mock_workspace()
        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [(session, None)]  # null agent name
        mock_db.execute.return_value = mock_result
        mock_db.get.return_value = session

        from app.api.v2.routers import active_calls as ac_module

        mini = FastAPI()
        register_exception_handlers(mini)
        mini.include_router(ac_module.router)
        mini.dependency_overrides[get_workspace] = lambda: ws
        mini.dependency_overrides[get_db] = lambda: mock_db

        client = TestClient(mini, raise_server_exceptions=False)
        resp = client.get("/active-calls")
        assert resp.status_code == 200
        assert resp.json()[0]["agent_name"] == "Unknown Agent"


# ── GET /active-calls/{room_name}/insights ────────────────────────────────────

_VALID_GEMINI_JSON = {
    "sentiment": "positive",
    "sentiment_score": 0.85,
    "topics": ["pricing", "features"],
    "summary": "Customer enquired about pricing. Agent provided clear answers.",
    "suggestions": ["Follow up with a quote.", "Send product brochure."],
}

_EXPECTED_RESPONSE_KEYS = {
    "sentiment", "sentiment_score", "topics", "summary", "suggestions", "call_duration_seconds"
}


class TestCallInsights:
    def _client(self, *, ws_id: uuid.UUID = WORKSPACE_ID) -> TestClient:
        return _build_app(db_sessions=[_make_session()], ws_id=ws_id)

    @patch("app.api.v2.routers.active_calls._enforce_insights_rate_limit", new_callable=AsyncMock)
    @patch("app.api.v2.routers.active_calls._read_redis_transcript", new_callable=AsyncMock)
    @patch("app.api.v2.routers.active_calls.gemini_service")
    def test_happy_path_correct_shape(self, mock_gemini, mock_redis, mock_rl):
        mock_redis.return_value = [
            {"role": "client", "text": "What does it cost?"},
            {"role": "agent", "text": "It starts at $99/month."},
        ]
        mock_gemini.generate_json.return_value = _VALID_GEMINI_JSON
        mock_rl.return_value = None

        client = self._client()
        resp = client.get(f"/active-calls/{ROOM_NAME}/insights")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == _EXPECTED_RESPONSE_KEYS
        assert body["sentiment"] == "positive"
        assert body["sentiment_score"] == 0.85
        assert body["topics"] == ["pricing", "features"]
        assert "pricing" in body["summary"].lower() or len(body["summary"]) > 0
        assert len(body["suggestions"]) == 2
        assert body["call_duration_seconds"] >= 0

    @patch("app.api.v2.routers.active_calls._enforce_insights_rate_limit", new_callable=AsyncMock)
    @patch("app.api.v2.routers.active_calls._read_redis_transcript", new_callable=AsyncMock)
    @patch("app.api.v2.routers.active_calls.gemini_service")
    def test_gemini_error_returns_fallback(self, mock_gemini, mock_redis, mock_rl):
        mock_redis.return_value = [{"role": "client", "text": "Hello"}]
        mock_gemini.generate_json.side_effect = Exception("Gemini quota exceeded")
        mock_rl.return_value = None

        client = self._client()
        resp = client.get(f"/active-calls/{ROOM_NAME}/insights")
        assert resp.status_code == 200
        body = resp.json()
        assert body["sentiment"] == "neutral"
        assert body["sentiment_score"] == 0.5
        assert body["topics"] == []
        assert body["summary"] == "Analysis unavailable."
        assert body["suggestions"] == []

    @patch("app.api.v2.routers.active_calls._enforce_insights_rate_limit", new_callable=AsyncMock)
    @patch("app.api.v2.routers.active_calls._read_redis_transcript", new_callable=AsyncMock)
    def test_no_transcript_returns_fallback(self, mock_redis, mock_rl):
        mock_redis.return_value = []
        mock_rl.return_value = None

        client = self._client()
        resp = client.get(f"/active-calls/{ROOM_NAME}/insights")
        assert resp.status_code == 200
        body = resp.json()
        assert body["sentiment"] == "neutral"
        assert body["summary"] == "Analysis unavailable."

    @patch("app.api.v2.routers.active_calls._enforce_insights_rate_limit", new_callable=AsyncMock)
    @patch("app.api.v2.routers.active_calls._read_redis_transcript", new_callable=AsyncMock)
    @patch("app.api.v2.routers.active_calls.gemini_service")
    def test_malformed_json_from_gemini_returns_fallback(self, mock_gemini, mock_redis, mock_rl):
        mock_redis.return_value = [{"role": "client", "text": "Hi"}]
        # generate_json raises json.JSONDecodeError when response is not parseable
        mock_gemini.generate_json.side_effect = json.JSONDecodeError("err", "", 0)
        mock_rl.return_value = None

        client = self._client()
        resp = client.get(f"/active-calls/{ROOM_NAME}/insights")
        assert resp.status_code == 200
        assert resp.json()["sentiment"] == "neutral"

    @patch("app.api.v2.routers.active_calls._enforce_insights_rate_limit", new_callable=AsyncMock)
    @patch("app.api.v2.routers.active_calls._read_redis_transcript", new_callable=AsyncMock)
    @patch("app.api.v2.routers.active_calls.gemini_service")
    def test_sentiment_score_clamped(self, mock_gemini, mock_redis, mock_rl):
        mock_redis.return_value = [{"role": "client", "text": "Hello"}]
        mock_gemini.generate_json.return_value = {
            **_VALID_GEMINI_JSON,
            "sentiment_score": 1.9,  # out of range — should clamp to 1.0
        }
        mock_rl.return_value = None

        client = self._client()
        resp = client.get(f"/active-calls/{ROOM_NAME}/insights")
        assert resp.status_code == 200
        assert resp.json()["sentiment_score"] == 1.0

    @patch("app.api.v2.routers.active_calls._enforce_insights_rate_limit", new_callable=AsyncMock)
    @patch("app.api.v2.routers.active_calls._read_redis_transcript", new_callable=AsyncMock)
    @patch("app.api.v2.routers.active_calls.gemini_service")
    def test_unknown_sentiment_normalised_to_neutral(self, mock_gemini, mock_redis, mock_rl):
        mock_redis.return_value = [{"role": "client", "text": "Hello"}]
        mock_gemini.generate_json.return_value = {
            **_VALID_GEMINI_JSON,
            "sentiment": "VERY_HAPPY",  # not in allowed set
        }
        mock_rl.return_value = None

        client = self._client()
        resp = client.get(f"/active-calls/{ROOM_NAME}/insights")
        assert resp.status_code == 200
        assert resp.json()["sentiment"] == "neutral"

    @patch("app.api.v2.routers.active_calls._read_redis_transcript", new_callable=AsyncMock)
    def test_workspace_isolation_returns_401(self, mock_redis):
        """Room belonging to a different workspace must return 401."""
        mock_redis.return_value = []

        # Client authenticates as OTHER_WORKSPACE, but session belongs to WORKSPACE_ID
        client = _build_app(
            db_sessions=[_make_session(tenant_id=WORKSPACE_ID)],
            ws_id=OTHER_WORKSPACE_ID,
        )
        resp = client.get(f"/active-calls/{ROOM_NAME}/insights")
        assert resp.status_code == 401

    async def _rate_limit_raiser(self, ws_id: str):
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail={})

    @patch("app.api.v2.routers.active_calls._enforce_insights_rate_limit")
    def test_rate_limit_returns_429(self, mock_rl):
        from fastapi import HTTPException, status

        async def _raise(_ws_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error": {"code": "rate_limit_exceeded"}},
            )

        mock_rl.side_effect = _raise
        client = self._client()
        resp = client.get(f"/active-calls/{ROOM_NAME}/insights")
        assert resp.status_code == 429

    @patch("app.api.v2.routers.active_calls._enforce_insights_rate_limit", new_callable=AsyncMock)
    def test_invalid_room_name_returns_400(self, mock_rl):
        mock_rl.return_value = None
        client = self._client()
        resp = client.get("/active-calls/not-a-valid-room/insights")
        assert resp.status_code == 400

    @patch("app.api.v2.routers.active_calls._enforce_insights_rate_limit", new_callable=AsyncMock)
    @patch("app.api.v2.routers.active_calls._read_redis_transcript", new_callable=AsyncMock)
    @patch("app.api.v2.routers.active_calls.gemini_service")
    def test_topics_capped_at_five(self, mock_gemini, mock_redis, mock_rl):
        mock_redis.return_value = [{"role": "client", "text": "Hello"}]
        mock_gemini.generate_json.return_value = {
            **_VALID_GEMINI_JSON,
            "topics": ["a", "b", "c", "d", "e", "f", "g"],  # 7 — should be trimmed to 5
        }
        mock_rl.return_value = None

        client = self._client()
        resp = client.get(f"/active-calls/{ROOM_NAME}/insights")
        assert resp.status_code == 200
        assert len(resp.json()["topics"]) == 5

    @patch("app.api.v2.routers.active_calls._enforce_insights_rate_limit", new_callable=AsyncMock)
    @patch("app.api.v2.routers.active_calls._read_redis_transcript", new_callable=AsyncMock)
    @patch("app.api.v2.routers.active_calls.gemini_service")
    def test_suggestions_capped_at_three(self, mock_gemini, mock_redis, mock_rl):
        mock_redis.return_value = [{"role": "client", "text": "Hello"}]
        mock_gemini.generate_json.return_value = {
            **_VALID_GEMINI_JSON,
            "suggestions": ["a", "b", "c", "d", "e"],  # 5 — should be trimmed to 3
        }
        mock_rl.return_value = None

        client = self._client()
        resp = client.get(f"/active-calls/{ROOM_NAME}/insights")
        assert resp.status_code == 200
        assert len(resp.json()["suggestions"]) == 3
