"""Unit tests for schema v1 gap requirements.

These tests verify ORM model attributes and migration metadata without
requiring a live database connection.
"""
from __future__ import annotations

import inspect

import sqlalchemy as sa
import pytest

from app.models.tenant import Tenant
from app.models.user import User
from app.models.api_key import Apikey
from app.models.invite import Invite
from app.models.refresh_token import RefreshToken


def _column(model, name: str) -> sa.Column:
    return model.__table__.c[name]


# ------------------------------------------------------------------ updated_at

class TestUpdatedAtPresent:
    def test_tenant_has_updated_at(self):
        col = _column(Tenant, "updated_at")
        assert isinstance(col.type, sa.DateTime)
        assert col.nullable is True

    def test_user_has_updated_at(self):
        col = _column(User, "updated_at")
        assert isinstance(col.type, sa.DateTime)
        assert col.nullable is True

    def test_apikey_has_updated_at(self):
        col = _column(Apikey, "updated_at")
        assert isinstance(col.type, sa.DateTime)
        assert col.nullable is True

    def test_invite_has_updated_at(self):
        col = _column(Invite, "updated_at")
        assert isinstance(col.type, sa.DateTime)
        assert col.nullable is True

    def test_refreshtoken_has_updated_at(self):
        col = _column(RefreshToken, "updated_at")
        assert isinstance(col.type, sa.DateTime)
        assert col.nullable is True


# ----------------------------------------------------------------- soft delete

class TestSoftDelete:
    def test_tenant_has_deleted_at(self):
        col = _column(Tenant, "deleted_at")
        assert isinstance(col.type, sa.DateTime)
        assert col.nullable is True

    def test_user_has_deleted_at(self):
        col = _column(User, "deleted_at")
        assert isinstance(col.type, sa.DateTime)
        assert col.nullable is True


# --------------------------------------------------------------- FK on_delete

class TestUserCurrentTenantFK:
    def test_current_tenant_id_has_set_null(self):
        col = _column(User, "current_tenant_id")
        fks = list(col.foreign_keys)
        assert len(fks) == 1
        fk = fks[0]
        assert fk.ondelete is not None
        assert fk.ondelete.upper() == "SET NULL"

    def test_current_tenant_id_is_nullable(self):
        col = _column(User, "current_tenant_id")
        assert col.nullable is True


# -------------------------------------------------------------------- indexes

class TestIndexes:
    def _index_names(self, model) -> set[str]:
        return {idx.name for idx in model.__table__.indexes}

    def test_invite_composite_index_exists(self):
        names = self._index_names(Invite)
        assert "ix_invite_email_tenant_id" in names

    def test_invite_composite_index_columns(self):
        idx = next(
            i for i in Invite.__table__.indexes
            if i.name == "ix_invite_email_tenant_id"
        )
        col_names = {c.name for c in idx.columns}
        assert col_names == {"email", "tenant_id"}

    def test_tenant_name_unique_active_index_exists(self):
        names = self._index_names(Tenant)
        assert "uq_tenant_name_active" in names

    def test_apikey_composite_index_exists(self):
        names = self._index_names(Apikey)
        assert "ix_apikey_key_hash_tenant_id" in names

    def test_user_email_unique(self):
        col = _column(User, "email")
        assert col.unique is True


# ------------------------------------------------------------------- created_at

class TestCreatedAt:
    @pytest.mark.parametrize("model", [Tenant, User, Apikey, Invite, RefreshToken])
    def test_created_at_has_server_default(self, model):
        col = _column(model, "created_at")
        assert col.server_default is not None


# ---------------------------------------------------------------- UUID primary keys

class TestUUIDPrimaryKeys:
    @pytest.mark.parametrize("model", [Tenant, User, Apikey, Invite, RefreshToken])
    def test_pk_is_uuid(self, model):
        pk_cols = [c for c in model.__table__.primary_key.columns]
        assert len(pk_cols) == 1
        assert isinstance(pk_cols[0].type, sa.UUID)


# --------------------------------------------------------- apikey key fields

class TestApikeyKeyFields:
    def test_key_hash_is_string64(self):
        col = _column(Apikey, "key_hash")
        assert isinstance(col.type, sa.String)
        assert col.type.length == 64

    def test_key_prefix_exists(self):
        col = _column(Apikey, "key_prefix")
        assert isinstance(col.type, sa.String)

    def test_key_hash_unique(self):
        col = _column(Apikey, "key_hash")
        assert col.unique is True


# ---------------------------------------------------------- migration file check

class TestMigrationFile:
    def test_gap_migration_exists(self):
        from pathlib import Path
        versions_dir = Path(__file__).parent.parent.parent / "alembic" / "versions"
        migration = versions_dir / "20260521_schema_v1_gaps.py"
        assert migration.exists(), "Gap migration file must exist"

    def test_gap_migration_has_correct_down_revision(self):
        import importlib.util
        from pathlib import Path
        path = (
            Path(__file__).parent.parent.parent
            / "alembic"
            / "versions"
            / "20260521_schema_v1_gaps.py"
        )
        spec = importlib.util.spec_from_file_location("gap_migration", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.down_revision == "20260518_tenant_name_uq"

    def test_gap_migration_has_upgrade_and_downgrade(self):
        import importlib.util
        from pathlib import Path
        path = (
            Path(__file__).parent.parent.parent
            / "alembic"
            / "versions"
            / "20260521_schema_v1_gaps.py"
        )
        spec = importlib.util.spec_from_file_location("gap_migration2", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert callable(mod.upgrade)
        assert callable(mod.downgrade)
