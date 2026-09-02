"""Unit tests for per-tenant daily LLM token budget tracking.

Covers app/services/token_budget_service.py:
  - record_daily_tokens: Redis pipeline INCRBY/INCR + TTL wiring, fail-open.
  - get_daily_usage / check_daily_budget: unlimited (None budget) vs. numeric
    budget, exact boundary behavior (two different comparisons for two
    different methods), and fail-open on Redis/DB errors.

Mocking convention for get_redis() matches tests/services/test_ghl_service.py
(patch app.services.<module>.get_redis, use AsyncMock for the redis client).
Tenant construction follows tests/services/test_api_key_service.py's
in-memory-SQLite `db` fixture convention.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.tenant import Tenant
from app.services.token_budget_service import token_budget_service


@pytest.fixture
def tenant(db):
    suffix = uuid.uuid4().hex[:8]
    t = Tenant(
        name=f"Budget Corp {suffix}",
        schema_name=f"budget_{suffix}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def tenant_with_budget(db):
    suffix = uuid.uuid4().hex[:8]
    t = Tenant(
        name=f"Budget Corp Capped {suffix}",
        schema_name=f"budget_capped_{suffix}",
        status="active",
        llm_token_budget_daily=500_000,
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


def _mock_pipeline(execute_results):
    """Build a mock redis pipeline supporting the fluent incrby/expire/incr
    chain used by record_daily_tokens, with .execute() resolving to the
    given results list."""
    pipe = MagicMock()
    pipe.incrby = MagicMock()
    pipe.expire = MagicMock()
    pipe.incr = MagicMock()
    pipe.execute = AsyncMock(return_value=execute_results)
    return pipe


class TestRecordDailyTokens:
    @pytest.mark.asyncio
    async def test_increments_pipeline_with_correct_keys_and_ttl(self):
        tenant_id = uuid.uuid4()
        mock_redis = MagicMock()
        pipe = _mock_pipeline([12345, 1])
        mock_redis.pipeline = MagicMock(return_value=pipe)

        with patch("app.services.token_budget_service.get_redis", return_value=mock_redis):
            result = await token_budget_service.record_daily_tokens(tenant_id, 100)

        assert result == 12345

        # INCRBY on the tokens key with the given token count.
        assert pipe.incrby.call_count == 1
        tokens_key_call = pipe.incrby.call_args
        tokens_key = tokens_key_call.args[0]
        assert tokens_key.startswith(f"tenant:{tenant_id}:tokens:")
        assert tokens_key_call.args[1] == 100

        # INCR on the calls key.
        assert pipe.incr.call_count == 1
        calls_key = pipe.incr.call_args.args[0]
        assert calls_key.startswith(f"tenant:{tenant_id}:calls:")

        # Both keys get a 172800s (48h) TTL via EXPIRE.
        assert pipe.expire.call_count == 2
        expire_calls = {c.args[0]: c.args[1] for c in pipe.expire.call_args_list}
        assert expire_calls[tokens_key] == 172800
        assert expire_calls[calls_key] == 172800

    @pytest.mark.asyncio
    async def test_returns_zero_when_redis_unavailable(self):
        tenant_id = uuid.uuid4()
        with patch("app.services.token_budget_service.get_redis", return_value=None):
            result = await token_budget_service.record_daily_tokens(tenant_id, 500)

        assert result == 0

    @pytest.mark.asyncio
    async def test_fails_open_on_redis_exception(self):
        tenant_id = uuid.uuid4()
        mock_redis = MagicMock()
        mock_redis.pipeline = MagicMock(side_effect=Exception("redis down"))

        with patch("app.services.token_budget_service.get_redis", return_value=mock_redis):
            # Must not raise.
            result = await token_budget_service.record_daily_tokens(tenant_id, 500)

        assert result == 0

    @pytest.mark.asyncio
    async def test_fails_open_when_pipeline_execute_raises(self):
        tenant_id = uuid.uuid4()
        mock_redis = MagicMock()
        pipe = MagicMock()
        pipe.incrby = MagicMock()
        pipe.expire = MagicMock()
        pipe.incr = MagicMock()
        pipe.execute = AsyncMock(side_effect=Exception("network blip"))
        mock_redis.pipeline = MagicMock(return_value=pipe)

        with patch("app.services.token_budget_service.get_redis", return_value=mock_redis):
            result = await token_budget_service.record_daily_tokens(tenant_id, 500)

        assert result == 0


class TestUnlimitedBudgetIsObserveOnly:
    """llm_token_budget_daily=None means unlimited/observe-only: usage is
    tracked but never blocks, regardless of how high it climbs."""

    @pytest.mark.asyncio
    async def test_get_daily_usage_never_exceeded_when_no_budget(self, db, tenant):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["10000000", "500"])

        with patch("app.services.token_budget_service.get_redis", return_value=mock_redis):
            usage = await token_budget_service.get_daily_usage(db, tenant.id)

        assert usage["budget_limit"] is None
        assert usage["total_tokens"] == 10_000_000
        assert usage["total_calls"] == 500
        assert usage["is_exceeded"] is False

    @pytest.mark.asyncio
    async def test_check_daily_budget_always_within_when_no_budget(self, db, tenant):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["10000000", "500"])

        with patch("app.services.token_budget_service.get_redis", return_value=mock_redis):
            within_budget, usage, limit = await token_budget_service.check_daily_budget(
                db, tenant.id
            )

        assert within_budget is True
        assert usage == 10_000_000
        assert limit is None


class TestNumericBudgetEnforcement:
    """tenant_with_budget has llm_token_budget_daily=500_000."""

    @pytest.mark.asyncio
    async def test_check_daily_budget_false_when_usage_exceeds_limit(self, db, tenant_with_budget):
        mock_redis = AsyncMock()
        # current_usage, then unused calls-count read
        mock_redis.get = AsyncMock(side_effect=["600000", "10"])

        with patch("app.services.token_budget_service.get_redis", return_value=mock_redis):
            within_budget, usage, limit = await token_budget_service.check_daily_budget(
                db, tenant_with_budget.id
            )

        assert within_budget is False
        assert usage == 600_000
        assert limit == 500_000

    @pytest.mark.asyncio
    async def test_check_daily_budget_true_when_usage_below_limit(self, db, tenant_with_budget):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["100000", "10"])

        with patch("app.services.token_budget_service.get_redis", return_value=mock_redis):
            within_budget, usage, limit = await token_budget_service.check_daily_budget(
                db, tenant_with_budget.id
            )

        assert within_budget is True
        assert usage == 100_000
        assert limit == 500_000

    @pytest.mark.asyncio
    async def test_check_daily_budget_boundary_usage_equals_limit_is_not_within_budget(
        self, db, tenant_with_budget
    ):
        """check_daily_budget's spec is `current_usage < budget` — at
        usage == limit exactly, that's False (blocked), unlike
        get_daily_usage's `>=` for is_exceeded which is also True here but
        via a different comparison operator."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["500000", "10"])

        with patch("app.services.token_budget_service.get_redis", return_value=mock_redis):
            within_budget, usage, limit = await token_budget_service.check_daily_budget(
                db, tenant_with_budget.id
            )

        assert within_budget is False
        assert usage == 500_000
        assert limit == 500_000

    @pytest.mark.asyncio
    async def test_get_daily_usage_boundary_usage_equals_limit_is_exceeded(
        self, db, tenant_with_budget
    ):
        """get_daily_usage's spec is `total_tokens >= budget_limit` for
        is_exceeded — at usage == limit exactly, that's True."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["500000", "10"])

        with patch("app.services.token_budget_service.get_redis", return_value=mock_redis):
            usage_dict = await token_budget_service.get_daily_usage(db, tenant_with_budget.id)

        assert usage_dict["is_exceeded"] is True
        assert usage_dict["total_tokens"] == 500_000
        assert usage_dict["budget_limit"] == 500_000

    @pytest.mark.asyncio
    async def test_get_daily_usage_not_exceeded_just_below_limit(self, db, tenant_with_budget):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["499999", "10"])

        with patch("app.services.token_budget_service.get_redis", return_value=mock_redis):
            usage_dict = await token_budget_service.get_daily_usage(db, tenant_with_budget.id)

        assert usage_dict["is_exceeded"] is False
        assert usage_dict["total_tokens"] == 499_999


class TestFailOpenBehavior:
    @pytest.mark.asyncio
    async def test_get_daily_usage_fails_open_when_redis_unavailable(self, db, tenant_with_budget):
        with patch("app.services.token_budget_service.get_redis", return_value=None):
            usage_dict = await token_budget_service.get_daily_usage(db, tenant_with_budget.id)

        assert usage_dict["total_tokens"] == 0
        assert usage_dict["total_calls"] == 0
        assert usage_dict["is_exceeded"] is False
        # Budget limit is still surfaced from the DB — only the Redis-backed
        # counters fail open to zero.
        assert usage_dict["budget_limit"] == 500_000

    @pytest.mark.asyncio
    async def test_get_daily_usage_fails_open_when_redis_raises(self, db, tenant_with_budget):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=Exception("redis down"))

        with patch("app.services.token_budget_service.get_redis", return_value=mock_redis):
            usage_dict = await token_budget_service.get_daily_usage(db, tenant_with_budget.id)

        assert usage_dict["total_tokens"] == 0
        assert usage_dict["total_calls"] == 0
        assert usage_dict["is_exceeded"] is False

    @pytest.mark.asyncio
    async def test_check_daily_budget_fails_open_when_redis_unavailable(
        self, db, tenant_with_budget
    ):
        with patch("app.services.token_budget_service.get_redis", return_value=None):
            within_budget, usage, limit = await token_budget_service.check_daily_budget(
                db, tenant_with_budget.id
            )

        # Redis read fails open to 0 usage; budget_limit is still resolved
        # from the DB and 0 < 500_000, so within_budget is True here for a
        # genuinely benign reason (no usage), not a masked failure.
        assert within_budget is True
        assert usage == 0
        assert limit == 500_000

    @pytest.mark.asyncio
    async def test_check_daily_budget_fails_open_on_db_lookup_exception(self, tenant_with_budget):
        """A broken DB session must never block a call. _get_budget_limit
        itself already fails open to None on a DB error, so check_daily_budget
        takes its normal "no budget configured" branch: usage is still read
        from Redis (unaffected by the DB failure), and the call is allowed
        through with budget_limit=None."""
        broken_db = MagicMock()
        broken_db.query = MagicMock(side_effect=Exception("db connection lost"))

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["999999", "10"])

        with patch("app.services.token_budget_service.get_redis", return_value=mock_redis):
            within_budget, usage, limit = await token_budget_service.check_daily_budget(
                broken_db, tenant_with_budget.id
            )

        assert within_budget is True
        assert usage == 999_999
        assert limit is None

    @pytest.mark.asyncio
    async def test_get_daily_usage_fails_open_on_db_lookup_exception(self):
        """_get_budget_limit itself catches DB errors and returns None —
        get_daily_usage as a whole must still return a well-formed dict."""
        broken_db = MagicMock()
        broken_db.query = MagicMock(side_effect=Exception("db connection lost"))
        tenant_id = uuid.uuid4()

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["1000", "5"])

        with patch("app.services.token_budget_service.get_redis", return_value=mock_redis):
            usage_dict = await token_budget_service.get_daily_usage(broken_db, tenant_id)

        assert usage_dict["budget_limit"] is None
        assert usage_dict["is_exceeded"] is False
        assert usage_dict["total_tokens"] == 1000
