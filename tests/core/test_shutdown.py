"""Tests for graceful shutdown resource cleanup."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from app.core.shutdown import graceful_shutdown


def test_graceful_shutdown_closes_resources():
    with patch("app.core.shutdown.close_rate_limiter", new_callable=AsyncMock) as close_rl:
        with patch(
            "app.middleware.api_key_middleware.close_auth_middleware_resources",
            new_callable=AsyncMock,
        ) as close_auth:
            asyncio.run(graceful_shutdown())
    close_rl.assert_awaited_once()
    close_auth.assert_awaited_once()
