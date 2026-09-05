"""Unit tests for the Redis-backed LLM circuit breaker.

Covers CLOSED/OPEN/HALF_OPEN transitions, cooldown-gated probing, and the
fail-open guarantees (disabled setting, missing Redis client, Redis errors)
that keep this breaker from ever blocking a live call turn.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.llm_circuit_breaker import (
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    LLMCircuitBreaker,
    _cooldown_key,
    _failures_key,
    _state_key,
)


@pytest.fixture
def breaker():
    return LLMCircuitBreaker()


@pytest.fixture
def mock_redis():
    return AsyncMock()


@pytest.fixture(autouse=True)
def enabled_breaker_settings(monkeypatch):
    """Deterministic defaults so tests don't depend on real env values."""
    from app.core.config import settings

    monkeypatch.setattr(settings.llm, "circuit_breaker_enabled", True)
    monkeypatch.setattr(settings.llm, "circuit_breaker_failure_threshold", 3)
    monkeypatch.setattr(settings.llm, "circuit_breaker_cooldown_sec", 30)
    return settings


class TestCanExecuteClosedState:
    @pytest.mark.asyncio
    async def test_closed_state_returns_true_without_touching_cooldown(
        self, breaker, mock_redis
    ):
        mock_redis.get.return_value = STATE_CLOSED

        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=mock_redis
        ):
            result = await breaker.can_execute("openai")

        assert result is True
        mock_redis.get.assert_awaited_once_with(_state_key("openai"))
        mock_redis.set.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_state_key_set_defaults_to_closed(self, breaker, mock_redis):
        mock_redis.get.return_value = None

        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=mock_redis
        ):
            result = await breaker.can_execute("openai")

        assert result is True
        mock_redis.set.assert_not_awaited()


class TestRecordFailure:
    @pytest.mark.asyncio
    async def test_increments_failure_counter(self, breaker, mock_redis):
        mock_redis.incr.return_value = 1

        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=mock_redis
        ):
            await breaker.record_failure("openai")

        mock_redis.incr.assert_awaited_once_with(_failures_key("openai"))

    @pytest.mark.asyncio
    async def test_below_threshold_does_not_trip_open(self, breaker, mock_redis):
        # threshold is 3 (autouse fixture); this is the 2nd failure.
        mock_redis.incr.return_value = 2

        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=mock_redis
        ):
            await breaker.record_failure("openai")

        mock_redis.set.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_threshold_reached_trips_open_and_sets_cooldown(
        self, breaker, mock_redis
    ):
        mock_redis.incr.return_value = 3

        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=mock_redis
        ):
            await breaker.record_failure("openai", error=RuntimeError("boom"))

        mock_redis.set.assert_any_call(_state_key("openai"), STATE_OPEN)
        mock_redis.set.assert_any_call(
            _cooldown_key("openai"), "cooling_down", ex=30
        )


class TestCanExecuteOpenState:
    @pytest.mark.asyncio
    async def test_open_within_cooldown_returns_false_fast(self, breaker, mock_redis):
        mock_redis.get.return_value = STATE_OPEN
        # cooldown key still present -> SET NX fails to acquire the probe.
        mock_redis.set.return_value = False

        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=mock_redis
        ):
            result = await breaker.can_execute("openai")

        assert result is False
        mock_redis.set.assert_awaited_once_with(
            _cooldown_key("openai"), "probe", nx=True, ex=30
        )

    @pytest.mark.asyncio
    async def test_cooldown_expired_probe_acquired_transitions_half_open(
        self, breaker, mock_redis
    ):
        mock_redis.get.return_value = STATE_OPEN
        # SET NX succeeds since the cooldown key is gone.
        mock_redis.set.return_value = True

        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=mock_redis
        ):
            result = await breaker.can_execute("openai")

        assert result is True
        mock_redis.set.assert_any_call(
            _cooldown_key("openai"), "probe", nx=True, ex=30
        )
        mock_redis.set.assert_any_call(_state_key("openai"), STATE_HALF_OPEN)

    @pytest.mark.asyncio
    async def test_probe_already_claimed_by_another_caller_returns_false(
        self, breaker, mock_redis
    ):
        mock_redis.get.return_value = STATE_OPEN
        mock_redis.set.return_value = False

        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=mock_redis
        ):
            result = await breaker.can_execute("openai")

        assert result is False
        # No state transition should be attempted when the probe wasn't acquired.
        state_calls = [
            call
            for call in mock_redis.set.await_args_list
            if call.args and call.args[0] == _state_key("openai")
        ]
        assert state_calls == []


class TestRecordSuccess:
    @pytest.mark.asyncio
    async def test_resets_counter_state_and_deletes_cooldown(self, breaker, mock_redis):
        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=mock_redis
        ):
            await breaker.record_success("openai")

        mock_redis.set.assert_any_call(_failures_key("openai"), 0)
        mock_redis.set.assert_any_call(_state_key("openai"), STATE_CLOSED)
        mock_redis.delete.assert_awaited_once_with(_cooldown_key("openai"))


class TestFailOpenOnMissingRedisClient:
    @pytest.mark.asyncio
    async def test_can_execute_true_when_redis_unavailable(self, breaker):
        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=None
        ):
            result = await breaker.can_execute("openai")

        assert result is True

    @pytest.mark.asyncio
    async def test_record_success_noop_when_redis_unavailable(self, breaker):
        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=None
        ):
            # Should not raise.
            await breaker.record_success("openai")

    @pytest.mark.asyncio
    async def test_record_failure_noop_when_redis_unavailable(self, breaker):
        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=None
        ):
            # Should not raise.
            await breaker.record_failure("openai", error=RuntimeError("boom"))


class TestFailOpenOnRedisError:
    @pytest.mark.asyncio
    async def test_can_execute_true_when_redis_raises(self, breaker, mock_redis):
        mock_redis.get.side_effect = ConnectionError("redis down")

        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=mock_redis
        ):
            result = await breaker.can_execute("openai")

        assert result is True

    @pytest.mark.asyncio
    async def test_record_success_swallows_redis_error(self, breaker, mock_redis):
        mock_redis.set.side_effect = ConnectionError("redis down")

        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=mock_redis
        ):
            # Should not raise.
            await breaker.record_success("openai")

    @pytest.mark.asyncio
    async def test_record_failure_swallows_redis_error(self, breaker, mock_redis):
        mock_redis.incr.side_effect = ConnectionError("redis down")

        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=mock_redis
        ):
            # Should not raise.
            await breaker.record_failure("openai", error=RuntimeError("boom"))


class TestDisabledBreaker:
    @pytest.mark.asyncio
    async def test_can_execute_true_without_calling_redis(
        self, breaker, mock_redis, enabled_breaker_settings
    ):
        enabled_breaker_settings.llm.circuit_breaker_enabled = False

        with patch(
            "app.services.llm_circuit_breaker.get_redis", return_value=mock_redis
        ) as mock_get_redis:
            result = await breaker.can_execute("openai")

        assert result is True
        mock_get_redis.assert_not_called()
        mock_redis.get.assert_not_awaited()
