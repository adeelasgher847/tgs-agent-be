"""Live PostgreSQL integration test for schema v2 Alembic rollback safety.

Ticket requirement: apply v1-state → apply v2 → rollback v2 → confirm v1 data intact.

Runs only when ``TEST_MIGRATION_DATABASE_URL`` (or a ``postgresql`` ``DATABASE_URL``)
points at a disposable database.  Skipped in CI/SQLite-only runs.

Usage::

    export TEST_MIGRATION_DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/tgs_migration_test
    export ELEVENLABS_ENCRYPTION_KEY=test-key-for-migration
    pytest tests/db/test_schema_v2_alembic_integration.py -v -m integration
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

# Revision immediately before schema v2 completion (parent of 20260602_schema_v2).
_V2_PARENT_REVISION = "20260602_phonenumber_provider"
_V2_REVISION = "20260602_schema_v2"


def _migration_database_url() -> str | None:
    explicit = (os.environ.get("TEST_MIGRATION_DATABASE_URL") or "").strip()
    if explicit:
        return explicit
    fallback = (os.environ.get("DATABASE_URL") or "").strip()
    if fallback.startswith("postgresql"):
        return fallback
    return None


def _alembic_config(db_url: str) -> Config:
    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


@pytest.fixture(scope="module")
def migration_db_url() -> str:
    url = _migration_database_url()
    if not url:
        pytest.skip(
            "Set TEST_MIGRATION_DATABASE_URL (postgresql) to run Alembic integration tests"
        )
    return url


@pytest.fixture(scope="module")
def alembic_cfg(migration_db_url: str) -> Config:
    os.environ.setdefault(
        "ELEVENLABS_ENCRYPTION_KEY",
        "test-migration-elevenlabs-encryption-key",
    )
    return _alembic_config(migration_db_url)


@pytest.fixture(scope="module")
def migration_engine(migration_db_url: str):
    engine = create_engine(migration_db_url)
    yield engine
    engine.dispose()


@pytest.mark.integration
class TestSchemaV2AlembicRollback:
    """v1-state → upgrade v2 → downgrade v2 → data intact → upgrade head."""

    def test_v2_upgrade_downgrade_preserves_core_rows(
        self,
        alembic_cfg: Config,
        migration_engine,
    ) -> None:
        tenant_id = uuid.uuid4()
        tenant_name = f"migration-test-{uuid.uuid4().hex[:12]}"

        # 1. Baseline: schema at parent of v2 (all v1 + callflow tables, no v2 completion).
        command.downgrade(alembic_cfg, _V2_PARENT_REVISION)

        with migration_engine.begin() as conn:
            conn.execute(
                text(
                    'INSERT INTO tenant (id, name, schema_name, status) '
                    "VALUES (:id, :name, :schema, 'active')"
                ),
                {
                    "id": str(tenant_id),
                    "name": tenant_name,
                    "schema": f"schema_{uuid.uuid4().hex[:8]}",
                },
            )

        with migration_engine.connect() as conn:
            row = conn.execute(
                text("SELECT name FROM tenant WHERE id = :id"),
                {"id": str(tenant_id)},
            ).one()
            assert row.name == tenant_name

        # 2. Apply v2 completion migration.
        command.upgrade(alembic_cfg, _V2_REVISION)

        with migration_engine.connect() as conn:
            smart_col = conn.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = 'agent' "
                    "AND column_name = 'smart_callback')"
                )
            ).scalar()
            assert smart_col is True

            ck_status = conn.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.table_constraints "
                    "WHERE table_schema = 'public' AND table_name = 'agent' "
                    "AND constraint_name = 'ck_agent_status_v2')"
                )
            ).scalar()
            assert ck_status is True

        # 3. Rollback v2 only — tenant row must survive unchanged.
        command.downgrade(alembic_cfg, _V2_PARENT_REVISION)

        with migration_engine.connect() as conn:
            row = conn.execute(
                text("SELECT name FROM tenant WHERE id = :id"),
                {"id": str(tenant_id)},
            ).one()
            assert row.name == tenant_name

            smart_col = conn.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = 'agent' "
                    "AND column_name = 'smart_callback')"
                )
            ).scalar()
            assert smart_col is False

        # 4. Re-apply v2 (idempotent path) and leave DB at head for other tests.
        command.upgrade(alembic_cfg, "head")

        with migration_engine.connect() as conn:
            row = conn.execute(
                text("SELECT name FROM tenant WHERE id = :id"),
                {"id": str(tenant_id)},
            ).one()
            assert row.name == tenant_name

        # Cleanup fixture tenant.
        with migration_engine.begin() as conn:
            conn.execute(
                text("DELETE FROM tenant WHERE id = :id"),
                {"id": str(tenant_id)},
            )


@pytest.mark.integration
class TestByoPgcryptoOnPostgres:
    """BYO key storage uses real pgp_sym_encrypt (not SQLite JWT monkeypatch)."""

    def test_encrypt_produces_pgcrypto_ciphertext(self, migration_engine) -> None:
        from sqlalchemy.orm import Session

        from app.core.db_encryption import encrypt_elevenlabs_key, is_pgcrypto_ciphertext

        with Session(bind=migration_engine) as db:
            ciphertext = encrypt_elevenlabs_key("xi-integration-test-key", db)

        assert ciphertext
        assert not ciphertext.startswith("eyJ")
        assert is_pgcrypto_ciphertext(ciphertext)
