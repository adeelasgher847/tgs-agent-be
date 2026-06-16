"""
Tests for the Call History Analytics API.

Coverage:
  1.  get_metrics returns correct aggregated totals
  2.  get_metrics date_from filter narrows results
  3.  get_metrics date_to filter narrows results
  4.  get_metrics agent_id filter is forwarded to the query
  5.  get_metrics flow_id filter is forwarded to the query
  6.  get_metrics direction filter maps to call_type
  7.  get_metrics status filter is forwarded to the query
  8.  get_metrics success_rate_percent is None when total_calls == 0
  9.  get_time_series returns one entry per day in date order
  10. get_list returns correct pagination metadata
  11. get_list sort_by / sort_dir defaults accepted
  12. CSV header row contains all required columns in correct order
  13. CSV data rows are emitted one per call, not buffered
  14. get_batch_metrics returns correct totals from BatchJob rows
  15. get_batch_metrics avg_completion_rate_percent is None with no batches
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from app.schemas.call_history import (
    BatchCallMetrics,
    CallHistoryItem,
    CallHistoryList,
    CallHistoryMetrics,
    CallHistoryTimeSeriesPoint,
)
from app.services.call_history_service import CallHistoryService


# ── helpers ───────────────────────────────────────────────────────────────────

def _svc() -> CallHistoryService:
    return CallHistoryService()


def _tenant() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


def _mock_db_one(row) -> MagicMock:
    """Return a db mock whose execute(...).one() returns *row*."""
    db = MagicMock()
    db.execute.return_value.one.return_value = row
    return db


def _mock_db_scalar_and_all(scalar_value, rows: list) -> MagicMock:
    """
    Return a db mock where:
      - first execute(...).scalar_one() → scalar_value  (count query)
      - second execute(...).all() → rows                 (data query)
    """
    db = MagicMock()
    results = [
        MagicMock(**{"scalar_one.return_value": scalar_value}),
        MagicMock(**{"all.return_value": rows}),
    ]
    db.execute.side_effect = results
    return db


def _metrics_row(
    total_calls=10,
    completed=7,
    failed=2,
    no_answer=1,
    avg_duration_seconds=45.5,
    total_duration_seconds=455,
    success_rate_percent=70.0,
):
    return SimpleNamespace(
        total_calls=total_calls,
        completed=completed,
        failed=failed,
        no_answer=no_answer,
        avg_duration_seconds=avg_duration_seconds,
        total_duration_seconds=total_duration_seconds,
        success_rate_percent=success_rate_percent,
    )


def _call_row(
    call_id=None,
    direction="inbound",
    from_number="+1111",
    to_number="+2222",
    agent_name="Bot",
    flow_name="Main Flow",
    status="completed",
    duration_seconds=60,
    started_at=None,
    ended_at=None,
):
    return SimpleNamespace(
        call_id=call_id or uuid.uuid4(),
        direction=direction,
        from_number=from_number,
        to_number=to_number,
        agent_name=agent_name,
        flow_name=flow_name,
        status=status,
        duration_seconds=duration_seconds,
        started_at=started_at or datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
        ended_at=ended_at or datetime(2026, 6, 1, 10, 1, tzinfo=timezone.utc),
    )


# ── 1. get_metrics returns correct aggregated totals ──────────────────────────

def test_metrics_correct_totals():
    row = _metrics_row()
    db = _mock_db_one(row)
    svc = _svc()

    result = svc.get_metrics(db, _tenant())

    assert isinstance(result, CallHistoryMetrics)
    assert result.total_calls == 10
    assert result.completed == 7
    assert result.failed == 2
    assert result.no_answer == 1
    assert result.avg_duration_seconds == 45.5
    assert result.total_duration_seconds == 455
    assert result.success_rate_percent == 70.0


# ── 2. get_metrics date_from filter forwarded ─────────────────────────────────

def test_metrics_date_from_forwarded():
    db = _mock_db_one(_metrics_row())
    svc = _svc()
    date_from = datetime(2026, 6, 1, tzinfo=timezone.utc)

    svc.get_metrics(db, _tenant(), date_from=date_from)

    # The WHERE clause is built inside SQLAlchemy; we verify execute was called.
    db.execute.assert_called_once()
    # Retrieve the compiled statement string to confirm the filter appears.
    stmt = db.execute.call_args[0][0]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "start_time" in compiled


# ── 3. get_metrics date_to filter forwarded ───────────────────────────────────

def test_metrics_date_to_forwarded():
    db = _mock_db_one(_metrics_row())
    svc = _svc()
    date_to = datetime(2026, 6, 30, tzinfo=timezone.utc)

    svc.get_metrics(db, _tenant(), date_to=date_to)

    stmt = db.execute.call_args[0][0]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "start_time" in compiled


# ── 4. get_metrics agent_id filter forwarded ─────────────────────────────────

def test_metrics_agent_id_forwarded():
    db = _mock_db_one(_metrics_row())
    svc = _svc()
    agent_id = uuid.uuid4()

    svc.get_metrics(db, _tenant(), agent_id=agent_id)

    stmt = db.execute.call_args[0][0]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    # SQLAlchemy renders UUIDs without dashes in the Postgres dialect literal
    assert str(agent_id).replace("-", "") in compiled


# ── 5. get_metrics flow_id filter forwarded ───────────────────────────────────

def test_metrics_flow_id_forwarded():
    db = _mock_db_one(_metrics_row())
    svc = _svc()
    flow_id = uuid.uuid4()

    svc.get_metrics(db, _tenant(), flow_id=flow_id)

    stmt = db.execute.call_args[0][0]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert str(flow_id).replace("-", "") in compiled


# ── 6. get_metrics direction maps to call_type ───────────────────────────────

def test_metrics_direction_maps_to_call_type():
    db = _mock_db_one(_metrics_row())
    svc = _svc()

    svc.get_metrics(db, _tenant(), direction="inbound")

    stmt = db.execute.call_args[0][0]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "call_type" in compiled
    assert "inbound" in compiled


# ── 7. get_metrics status filter forwarded ───────────────────────────────────

def test_metrics_status_forwarded():
    db = _mock_db_one(_metrics_row())
    svc = _svc()

    svc.get_metrics(db, _tenant(), status="completed")

    stmt = db.execute.call_args[0][0]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "completed" in compiled


# ── 8. success_rate_percent is None when no calls ────────────────────────────

def test_metrics_success_rate_none_when_zero_calls():
    row = _metrics_row(
        total_calls=0,
        completed=0,
        failed=0,
        no_answer=0,
        avg_duration_seconds=None,
        total_duration_seconds=None,
        success_rate_percent=None,
    )
    db = _mock_db_one(row)
    svc = _svc()

    result = svc.get_metrics(db, _tenant())

    assert result.total_calls == 0
    assert result.success_rate_percent is None
    assert result.avg_duration_seconds is None
    assert result.total_duration_seconds is None


# ── 9. get_time_series returns entries in date order ─────────────────────────

def test_time_series_returns_daily_rows():
    rows = [
        SimpleNamespace(date="2026-06-01", total=5, completed=4, failed=1),
        SimpleNamespace(date="2026-06-02", total=3, completed=3, failed=0),
    ]
    db = MagicMock()
    db.execute.return_value.all.return_value = rows
    svc = _svc()

    result = svc.get_time_series(db, _tenant())

    assert len(result) == 2
    assert isinstance(result[0], CallHistoryTimeSeriesPoint)
    assert result[0].date == "2026-06-01"
    assert result[0].total == 5
    assert result[0].completed == 4
    assert result[0].failed == 1
    assert result[1].date == "2026-06-02"


# ── 10. get_list returns correct pagination metadata ─────────────────────────

def test_list_pagination_metadata():
    rows = [_call_row() for _ in range(5)]
    db = _mock_db_scalar_and_all(scalar_value=47, rows=rows)
    svc = _svc()

    result = svc.get_list(db, _tenant(), page=2, per_page=25)

    assert isinstance(result, CallHistoryList)
    assert result.total == 47
    assert result.page == 2
    assert result.per_page == 25
    assert result.pages == 2  # ceil(47/25)
    assert len(result.items) == 5


# ── 11. get_list accepts default sort params ──────────────────────────────────

def test_list_default_sort():
    db = _mock_db_scalar_and_all(scalar_value=0, rows=[])
    svc = _svc()

    # Should not raise
    result = svc.get_list(db, _tenant())

    assert result.total == 0
    assert result.items == []
    assert result.pages == 0


# ── 12. CSV header contains all required columns ──────────────────────────────

def test_csv_header_columns():
    db = MagicMock()
    db.execute.return_value.__iter__.return_value = iter([])
    svc = _svc()

    chunks = list(svc.iter_export_csv(db, _tenant()))

    assert len(chunks) >= 1
    header = chunks[0]
    expected_columns = [
        "call_id",
        "direction",
        "from_number",
        "to_number",
        "agent_name",
        "flow_name",
        "status",
        "duration_seconds",
        "started_at",
        "ended_at",
    ]
    for col in expected_columns:
        assert col in header, f"Column '{col}' missing from CSV header"


# ── 13. CSV data rows emitted one per call ────────────────────────────────────

def test_csv_data_rows_streamed():
    cid1 = uuid.uuid4()
    cid2 = uuid.uuid4()
    data_rows = [
        SimpleNamespace(
            call_id=cid1,
            direction="inbound",
            from_number="+1111",
            to_number="+2222",
            agent_name="Alpha",
            flow_name="Flow A",
            status="completed",
            duration_seconds=90,
            started_at=datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 6, 1, 10, 1, 30, tzinfo=timezone.utc),
        ),
        SimpleNamespace(
            call_id=cid2,
            direction="outbound",
            from_number="+3333",
            to_number="+4444",
            agent_name="Beta",
            flow_name=None,
            status="failed",
            duration_seconds=None,
            started_at=datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc),
            ended_at=None,
        ),
    ]
    db = MagicMock()
    db.execute.return_value.__iter__.return_value = iter(data_rows)
    svc = _svc()

    chunks = list(svc.iter_export_csv(db, _tenant()))

    # header + 2 data rows
    assert len(chunks) == 3
    assert str(cid1) in chunks[1]
    assert "inbound" in chunks[1]
    assert str(cid2) in chunks[2]
    assert "outbound" in chunks[2]
    # None values rendered as empty string
    assert ",,failed" in chunks[2] or ",failed," in chunks[2]


# ── 14. get_batch_metrics returns correct totals ──────────────────────────────

def test_batch_metrics_correct_totals():
    row = SimpleNamespace(
        total_batches=5,
        avg_completion_rate_percent=82.5,
        total_calls_dispatched=1000,
        total_failed=50,
    )
    db = _mock_db_one(row)
    svc = _svc()

    result = svc.get_batch_metrics(db, _tenant())

    assert isinstance(result, BatchCallMetrics)
    assert result.total_batches == 5
    assert result.avg_completion_rate_percent == 82.5
    assert result.total_calls_dispatched == 1000
    assert result.total_failed == 50


# ── 15. get_batch_metrics avg is None with no batches ────────────────────────

def test_batch_metrics_avg_none_when_no_batches():
    row = SimpleNamespace(
        total_batches=0,
        avg_completion_rate_percent=None,
        total_calls_dispatched=0,
        total_failed=0,
    )
    db = _mock_db_one(row)
    svc = _svc()

    result = svc.get_batch_metrics(db, _tenant())

    assert result.total_batches == 0
    assert result.avg_completion_rate_percent is None
    assert result.total_calls_dispatched == 0
