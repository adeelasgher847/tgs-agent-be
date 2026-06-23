"""Unit tests for Schema v2 gap requirements.

These tests verify ORM model attributes and migration metadata without
requiring a live database connection (same pattern as test_schema_v1_gaps.py).
"""
from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import sqlalchemy as sa
import pytest

from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.prompt_version import PromptVersion
from app.models.folder import Folder
from app.models.folder_flow import FolderFlow


# ──────────────────────────────────────────────────── helpers ─────────────────

def _column(model, name: str) -> sa.Column:
    return model.__table__.c[name]


def _index_names(model) -> set[str]:
    return {idx.name for idx in model.__table__.indexes}


def _constraint_names(model) -> set[str]:
    return {c.name for c in model.__table__.constraints}


def _check_constraint(model, name: str):
    for c in model.__table__.constraints:
        if isinstance(c, sa.CheckConstraint) and c.name == name:
            return c
    return None


# ─────────────────────────────────────────── agent model ─────────────────────

class TestAgentV2Columns:
    def test_smart_callback_column_exists(self):
        col = _column(Agent, "smart_callback")
        assert isinstance(col.type, sa.Boolean)

    def test_smart_callback_not_nullable(self):
        col = _column(Agent, "smart_callback")
        assert col.nullable is False

    def test_smart_callback_server_default_false(self):
        col = _column(Agent, "smart_callback")
        assert col.server_default is not None

    def test_encrypted_elevenlabs_api_key_exists(self):
        col = _column(Agent, "encrypted_elevenlabs_api_key")
        assert isinstance(col.type, sa.Text)
        assert col.nullable is True

    def test_status_column_exists(self):
        col = _column(Agent, "status")
        assert isinstance(col.type, sa.String)

    def test_llm_model_column_nullable(self):
        col = _column(Agent, "llm_model")
        assert col.nullable is True


class TestAgentV2Indexes:
    def test_ix_agent_tenant_id_exists(self):
        names = _index_names(Agent)
        assert "ix_agent_tenant_id" in names


# ────────────────────────────────────────── callflow model ───────────────────

class TestCallFlowV2:
    def test_ix_callflow_agent_id_exists(self):
        names = _index_names(CallFlow)
        assert "ix_callflow_agent_id" in names

    def test_direction_check_includes_bidirectional(self):
        ck = _check_constraint(CallFlow, "ck_callflow_direction")
        assert ck is not None, "ck_callflow_direction constraint must exist on CallFlow"
        assert "bidirectional" in str(ck.sqltext)

    def test_direction_check_retains_inbound_outbound(self):
        ck = _check_constraint(CallFlow, "ck_callflow_direction")
        assert ck is not None
        text = str(ck.sqltext)
        assert "inbound" in text
        assert "outbound" in text

    def test_welcome_message_type_check_exists(self):
        ck = _check_constraint(CallFlow, "ck_callflow_welcome_message_type")
        assert ck is not None, "ck_callflow_welcome_message_type must exist on CallFlow"

    def test_welcome_message_type_check_values(self):
        ck = _check_constraint(CallFlow, "ck_callflow_welcome_message_type")
        text = str(ck.sqltext)
        assert "user_initiated" in text
        assert "ai_dynamic" in text
        assert "ai_custom" in text

    def test_is_deleted_exists(self):
        col = _column(CallFlow, "is_deleted")
        assert isinstance(col.type, sa.Boolean)
        assert col.nullable is False


# ──────────────────────────────────────── promptversion model ────────────────

class TestPromptVersionV2:
    def test_flow_id_index_exists(self):
        names = _index_names(PromptVersion)
        # Accept either simple ix_promptversion_flow_id OR composite ix_promptversion_flow_created
        has_flow_index = (
            "ix_promptversion_flow_id" in names
            or "ix_promptversion_flow_created" in names
        )
        assert has_flow_index, "PromptVersion must have an index covering flow_id"

    def test_prompt_text_not_nullable(self):
        col = _column(PromptVersion, "prompt_text")
        assert col.nullable is False

    def test_created_at_has_server_default(self):
        col = _column(PromptVersion, "created_at")
        assert col.server_default is not None


# ─────────────────────────────────────────── folder model ────────────────────

class TestFolderV2:
    def test_is_deleted_exists(self):
        col = _column(Folder, "is_deleted")
        assert isinstance(col.type, sa.Boolean)

    def test_tenant_id_index_exists(self):
        names = _index_names(Folder)
        assert "ix_folder_tenant_id" in names


# ────────────────────────────────────────── folderflow model ─────────────────

class TestFolderFlowV2:
    def test_unique_folder_flow_constraint_exists(self):
        names = _constraint_names(FolderFlow)
        assert "uq_folderflow_folder_flow" in names

    def test_has_surrogate_id(self):
        col = _column(FolderFlow, "id")
        assert isinstance(col.type, sa.UUID)


# ────────────────────────────────────── agent schema enum ────────────────────

class TestAgentCreateSchemaDefaults:
    def test_create_default_status_is_pending(self):
        from app.schemas.agent import (
            AgentCreate,
            AgentStatusEnum,
            LanguageEnum,
            TtsModelSchema,
            TtsProviderEnum,
        )

        body = AgentCreate(
            name="Pending Default Agent",
            llm_model="gpt-4o-mini",
            tts_model=TtsModelSchema(
                provider=TtsProviderEnum.elevenlabs,
                voice_id="voice-1",
                language=LanguageEnum.en,
            ),
        )
        assert body.status == AgentStatusEnum.pending


class TestAgentOutStatusFallback:
    def test_unknown_status_defaults_to_pending(self):
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from app.schemas.agent import AgentStatusEnum, agent_to_out

        agent = MagicMock()
        agent.id = uuid.uuid4()
        agent.name = "Unknown Status Agent"
        agent.llm_model = "gpt-4o-mini"
        agent.tts_provider_slug = None
        agent.tts_voice_external_id = None
        agent.tts_language = None
        agent.status = "not_a_real_status"
        agent.created_at = datetime.now(timezone.utc)
        agent.updated_at = None
        agent.tenant_id = uuid.uuid4()
        agent.system_prompt = None
        agent.language = None
        agent.voice_type = None
        agent.is_inbound_agent = False
        agent.is_follow_up_agent = False

        out = agent_to_out(agent)
        assert out.status == AgentStatusEnum.pending

    def test_null_status_defaults_to_pending(self):
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from app.schemas.agent import AgentStatusEnum, agent_to_out

        agent = MagicMock()
        agent.id = uuid.uuid4()
        agent.name = "Null Status Agent"
        agent.llm_model = None
        agent.tts_provider_slug = None
        agent.tts_voice_external_id = None
        agent.tts_language = None
        agent.status = None
        agent.created_at = datetime.now(timezone.utc)
        agent.updated_at = None
        agent.tenant_id = uuid.uuid4()
        agent.system_prompt = None
        agent.language = None
        agent.voice_type = None
        agent.is_inbound_agent = None
        agent.is_follow_up_agent = None

        assert agent_to_out(agent).status == AgentStatusEnum.pending


class TestAgentStatusEnum:
    def test_error_status_in_enum(self):
        from app.schemas.agent import AgentStatusEnum
        assert "error" in [e.value for e in AgentStatusEnum]

    def test_pending_status_in_enum(self):
        from app.schemas.agent import AgentStatusEnum
        assert "pending" in [e.value for e in AgentStatusEnum]

    def test_ready_status_in_enum(self):
        from app.schemas.agent import AgentStatusEnum
        assert "ready" in [e.value for e in AgentStatusEnum]

    def test_legacy_statuses_still_in_enum(self):
        from app.schemas.agent import AgentStatusEnum
        values = [e.value for e in AgentStatusEnum]
        assert "active" in values
        assert "inactive" in values
        assert "draft" in values


# ────────────────────────────────────── callflow schema ──────────────────────

class TestCallFlowSchema:
    def test_direction_enum_has_bidirectional(self):
        from app.schemas.call_flow import DirectionEnum
        assert "bidirectional" in [e.value for e in DirectionEnum]

    def test_direction_enum_retains_inbound_outbound(self):
        from app.schemas.call_flow import DirectionEnum
        values = [e.value for e in DirectionEnum]
        assert "inbound" in values
        assert "outbound" in values

    def test_welcome_message_type_enum_exists(self):
        from app.schemas.call_flow import WelcomeMessageTypeEnum
        values = [e.value for e in WelcomeMessageTypeEnum]
        assert "user_initiated" in values
        assert "ai_dynamic" in values
        assert "ai_custom" in values

    def test_call_flow_create_uses_welcome_message_type_enum(self):
        from app.schemas.call_flow import CallFlowCreate, WelcomeMessageTypeEnum
        import uuid
        flow = CallFlowCreate(
            name="test",
            direction="outbound",
            agentId=uuid.uuid4(),
            welcomeMessageType="ai_dynamic",
        )
        assert flow.welcome_message_type == WelcomeMessageTypeEnum.ai_dynamic

    def test_call_flow_create_rejects_invalid_welcome_message_type(self):
        import uuid
        from pydantic import ValidationError
        from app.schemas.call_flow import CallFlowCreate
        with pytest.raises(ValidationError):
            CallFlowCreate(
                name="test",
                direction="outbound",
                agentId=uuid.uuid4(),
                welcomeMessageType="invalid_type",
            )


# ─────────────────────────────────────────── llm models ──────────────────────

class TestLlmModelsV2:
    def test_ticket_models_in_allow_list(self):
        from app.core.llm_models import ALLOWED_LLM_MODELS
        ticket_models = [
            "gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini",
            "gemini-2.5-flash", "gemini-2.0-flash-001",
        ]
        for model in ticket_models:
            assert model in ALLOWED_LLM_MODELS, f"{model} missing from ALLOWED_LLM_MODELS"

    def test_no_space_in_gemini_flash(self):
        from app.core.llm_models import ALLOWED_LLM_MODELS
        # Ticket had a typo "gemini-2.5- flash" — ensure no space variant is present
        assert "gemini-2.5- flash" not in ALLOWED_LLM_MODELS


# ───────────────────────────────────────── db_encryption ─────────────────────

class TestDbEncryptionModule:
    def test_encrypt_raises_without_key(self):
        from app.core.db_encryption import encrypt_elevenlabs_key
        mock_db = MagicMock()
        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.ELEVENLABS_ENCRYPTION_KEY = ""
            with pytest.raises(ValueError, match="ELEVENLABS_ENCRYPTION_KEY"):
                encrypt_elevenlabs_key("some-key", mock_db)

    def test_decrypt_raises_without_key(self):
        from app.core.db_encryption import decrypt_elevenlabs_key
        mock_db = MagicMock()
        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.ELEVENLABS_ENCRYPTION_KEY = ""
            with pytest.raises(ValueError, match="ELEVENLABS_ENCRYPTION_KEY"):
                decrypt_elevenlabs_key("ciphertext", mock_db)

    def test_encrypt_calls_pgcrypto_sql(self):
        from app.core.db_encryption import encrypt_elevenlabs_key
        mock_db = MagicMock()
        mock_db.execute.return_value.scalar.return_value = "base64ciphertext"
        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.ELEVENLABS_ENCRYPTION_KEY = "test-enc-key"
            result = encrypt_elevenlabs_key("plaintext", mock_db)
        assert result == "base64ciphertext"
        mock_db.execute.assert_called_once()
        # The first positional arg is the TextClause — inspect its .text attribute
        sql_clause = mock_db.execute.call_args[0][0]
        assert "pgp_sym_encrypt" in sql_clause.text

    def test_decrypt_calls_pgcrypto_sql(self):
        from app.core.db_encryption import decrypt_elevenlabs_key
        mock_db = MagicMock()
        mock_db.execute.return_value.scalar.return_value = "plaintext"
        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.ELEVENLABS_ENCRYPTION_KEY = "test-enc-key"
            result = decrypt_elevenlabs_key("base64ct", mock_db)
        assert result == "plaintext"
        mock_db.execute.assert_called_once()
        sql_clause = mock_db.execute.call_args[0][0]
        assert "pgp_sym_decrypt" in sql_clause.text

    def test_is_legacy_jwt_ciphertext_compact_jws(self):
        from app.core.db_encryption import is_legacy_jwt_ciphertext
        assert is_legacy_jwt_ciphertext("eyJhbGciOiJIUzI1NiJ9.payload.sig") is True
        assert is_legacy_jwt_ciphertext("eyJonly-one-part") is False

    def test_is_pgcrypto_ciphertext_rejects_jwt(self):
        from app.core.db_encryption import is_pgcrypto_ciphertext
        assert is_pgcrypto_ciphertext("eyJhbGciOiJIUzI1NiJ9.payload.sig") is False

    def test_is_pgcrypto_ciphertext_accepts_openpgp_marker(self):
        import base64

        from app.core.db_encryption import is_pgcrypto_ciphertext

        blob = base64.b64encode(bytes([0xC3, 0x03, 0x07, 0x04])).decode()
        assert is_pgcrypto_ciphertext(blob) is True

    def test_is_pgcrypto_ciphertext_rejects_garbage(self):
        from app.core.db_encryption import is_pgcrypto_ciphertext
        assert is_pgcrypto_ciphertext("not-valid-ciphertext!!!") is False

    def test_is_pgcrypto_ciphertext_empty(self):
        from app.core.db_encryption import is_pgcrypto_ciphertext
        assert is_pgcrypto_ciphertext("") is False

    def test_decrypt_stored_elevenlabs_key_rejects_unrecognized_format(self):
        from app.core.db_encryption import decrypt_stored_elevenlabs_key

        with pytest.raises(ValueError, match="Unrecognized ElevenLabs key"):
            decrypt_stored_elevenlabs_key("not-valid-ciphertext!!!", db=None)

    def test_decrypt_stored_elevenlabs_key_jwt_roundtrip(self):
        from app.core.db_encryption import decrypt_stored_elevenlabs_key
        from app.core.security import encrypt_api_key

        plaintext = "xi-test-by-key-roundtrip"
        token = encrypt_api_key(plaintext)
        assert token.startswith("eyJ")
        assert decrypt_stored_elevenlabs_key(token, db=None) == plaintext

    def test_encrypt_decrypt_roundtrip_via_mock_db(self):
        from app.core.db_encryption import (
            decrypt_stored_elevenlabs_key,
            encrypt_elevenlabs_key,
        )

        mock_db = MagicMock()
        stored: dict[str, str] = {}

        def _fake_execute(stmt, params=None):
            sql = stmt.text if hasattr(stmt, "text") else str(stmt)
            params = params or {}
            result = MagicMock()
            if "pgp_sym_encrypt" in sql:
                stored["ct"] = f"enc:{params.get('pt', '')}"
                result.scalar.return_value = stored["ct"]
            elif "pgp_sym_decrypt" in sql:
                ct = params.get("ct", "")
                if ct.startswith("enc:"):
                    result.scalar.return_value = ct[4:]
                else:
                    result.scalar.return_value = ""
            return result

        mock_db.execute.side_effect = _fake_execute
        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.ELEVENLABS_ENCRYPTION_KEY = "test-enc-key"
            ciphertext = encrypt_elevenlabs_key("roundtrip-secret", mock_db)
        assert not ciphertext.startswith("eyJ")
        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.ELEVENLABS_ENCRYPTION_KEY = "test-enc-key"
            with patch("app.core.db_encryption.is_pgcrypto_ciphertext", return_value=True):
                assert (
                    decrypt_stored_elevenlabs_key(ciphertext, db=mock_db)
                    == "roundtrip-secret"
                )


# ────────────────────────────────────── migration file checks ────────────────

_VERSIONS_DIR = Path(__file__).parent.parent.parent / "alembic" / "versions"


def _load_migration(filename: str):
    path = _VERSIONS_DIR / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestV2MigrationFile:
    def test_file_exists(self):
        assert (_VERSIONS_DIR / "20260602_schema_v2_completion.py").exists()

    def test_down_revision_is_phonenumber_provider(self):
        mod = _load_migration("20260602_schema_v2_completion.py")
        assert mod.down_revision == "20260602_phonenumber_provider"

    def test_has_upgrade_and_downgrade(self):
        mod = _load_migration("20260602_schema_v2_completion.py")
        assert callable(mod.upgrade)
        assert callable(mod.downgrade)

    def test_revision_id(self):
        mod = _load_migration("20260602_schema_v2_completion.py")
        assert mod.revision == "20260602_schema_v2"

    def test_llm_check_sql_uses_revision_snapshot(self):
        mod = _load_migration("20260602_schema_v2_completion.py")
        sql = mod._llm_check_sql()
        for model in mod._ALLOWED_LLM_MODELS_AT_REVISION:
            assert f"'{model}'" in sql

    def test_remigrate_fails_without_key_when_jwt_rows_exist(self):
        mod = _load_migration("20260602_schema_v2_completion.py")
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("agent-uuid", "eyJhbGciOiJIUzI1NiJ9.payload.sig"),
        ]
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.ELEVENLABS_ENCRYPTION_KEY = ""
            with pytest.raises(RuntimeError, match="ELEVENLABS_ENCRYPTION_KEY"):
                mod._remigrate_elevenlabs_keys(mock_conn)

    def test_remigrate_skips_when_no_jwt_rows_and_no_key(self):
        mod = _load_migration("20260602_schema_v2_completion.py")
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("agent-uuid", None),
        ]
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.ELEVENLABS_ENCRYPTION_KEY = ""
            mod._remigrate_elevenlabs_keys(mock_conn)  # must not raise

    def test_does_not_modify_v1_migration(self):
        """Confirm we have not altered the v1 gaps migration file."""
        mod = _load_migration("20260521_schema_v1_gaps.py")
        assert mod.down_revision == "20260518_tenant_name_uq"
