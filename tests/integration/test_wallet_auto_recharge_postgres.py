"""Wallet auto-recharge cooldown-claim concurrency, against real PostgreSQL.

`CreditService._maybe_trigger_wallet_auto_recharge`'s "only one charge fires
per cooldown window even under concurrent sessions" claim relies on real
READ COMMITTED row-lock semantics for its atomic
`UPDATE ... WHERE last_triggered_at IS NULL OR < cutoff` claim. The
SQLite-backed unit tests in `tests/services/test_credit_service.py` only
exercise this sequentially in a single thread/connection, so they trivially
pass regardless of whether the claim is actually safe under concurrency.

This test races two genuinely concurrent Postgres sessions (separate
threads, separate connections/transactions) against the same
`WalletAutoRechargeConfig` row's claim UPDATE and asserts exactly one wins
(`rowcount == 1`).

Requires TEST_DATABASE_URL. Skipped otherwise.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import or_
from sqlalchemy import update as sa_update

from app.models.tenant import Tenant
from app.models.wallet_auto_recharge_config import WalletAutoRechargeConfig
from tests.conftest import _INTEGRATION_SKIP

pytestmark = [_INTEGRATION_SKIP, pytest.mark.integration]


@pytest.fixture()
def tenant(pg_session):
    t = Tenant(
        name=f"PG Wallet Tenant {uuid.uuid4().hex[:8]}",
        schema_name=f"pg_wallet_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    pg_session.add(t)
    pg_session.commit()
    pg_session.refresh(t)
    return t


@pytest.fixture()
def auto_recharge_config(pg_session, tenant):
    config = WalletAutoRechargeConfig(
        workspace_id=tenant.id,
        enabled=True,
        min_balance=Decimal("8.00"),
        recharge_amount=Decimal("5.00"),
        stripe_payment_method_id="pm_test_concurrency",
    )
    pg_session.add(config)
    pg_session.commit()
    pg_session.refresh(config)
    return config


def _claim(
    pg_session_factory,
    config_id: uuid.UUID,
    now: datetime,
    cutoff: datetime,
    results: list,
    index: int,
):
    """Runs the exact same atomic claim UPDATE
    `CreditService._maybe_trigger_wallet_auto_recharge` issues, on its own
    dedicated session/connection, and records the rowcount it got."""
    session = pg_session_factory()
    try:
        claim_stmt = (
            sa_update(WalletAutoRechargeConfig)
            .where(
                WalletAutoRechargeConfig.id == config_id,
                WalletAutoRechargeConfig.enabled.is_(True),
                WalletAutoRechargeConfig.stripe_payment_method_id.isnot(None),
                or_(
                    WalletAutoRechargeConfig.last_triggered_at.is_(None),
                    WalletAutoRechargeConfig.last_triggered_at < cutoff,
                ),
            )
            .values(
                last_triggered_at=now,
                last_trigger_status="pending",
                last_payment_intent_id=None,
                last_trigger_error=None,
            )
        )
        result = session.execute(claim_stmt)
        session.commit()
        results[index] = result.rowcount
    finally:
        session.close()


def test_concurrent_claim_only_one_thread_wins(
    pg_session_factory, pg_session, auto_recharge_config
):
    """Two threads race the atomic cooldown-claim UPDATE on the same config
    row (last_triggered_at is NULL, so both threads' WHERE clause initially
    matches) — real Postgres row-lock/READ COMMITTED semantics must
    serialize them so exactly one gets rowcount == 1 and the other gets 0."""
    config_id = auto_recharge_config.id
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=300)

    results: list[int | None] = [None, None]
    barrier = threading.Barrier(2)

    def _worker(index: int):
        barrier.wait()  # maximize the chance both threads race the same window
        _claim(pg_session_factory, config_id, now, cutoff, results, index)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert None not in results, f"a claim thread did not complete: {results}"
    assert sorted(results) == [0, 1], (
        f"expected exactly one thread to claim the trigger (rowcount 1) and the "
        f"other to lose (rowcount 0), got {results}"
    )

    # Sanity: the row was actually updated to reflect the winning claim.
    pg_session.expire_all()
    refreshed = pg_session.get(WalletAutoRechargeConfig, config_id)
    assert refreshed.last_triggered_at is not None
    assert refreshed.last_trigger_status == "pending"


def test_concurrent_claim_respects_cooldown_already_claimed(
    pg_session_factory, pg_session, auto_recharge_config
):
    """If the row was already claimed recently (last_triggered_at within the
    cooldown window), a fresh concurrent claim attempt must never win."""
    config_id = auto_recharge_config.id
    now = datetime.now(timezone.utc)

    already_claimed_at = now - timedelta(seconds=30)  # well inside the 300s cooldown
    auto_recharge_config.last_triggered_at = already_claimed_at
    pg_session.commit()

    cutoff = now - timedelta(seconds=300)
    results: list[int | None] = [None, None]
    barrier = threading.Barrier(2)

    def _worker(index: int):
        barrier.wait()
        _claim(pg_session_factory, config_id, now, cutoff, results, index)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert results == [
        0,
        0,
    ], f"expected both claims to lose (still in cooldown), got {results}"
