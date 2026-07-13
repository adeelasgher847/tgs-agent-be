"""
Vapi-style intelligent contact recovery additions:
1) Deterministic email STT-artifact cleanup before strict validation.
2) Natural-confirmation path that promotes name_confident without spelling.
3) Public apply_post_call_recovery helper (upgrade-only).
4) Post-call LLM recovery wiring in PostCallAppointmentService.

These tests are additive — they exercise the new paths and verify that the
existing strict behavior is NOT regressed.
"""

import os
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.core.config import settings
from app.db.base import Base
from app.models.agent import Agent
from app.models.appointment import Appointment
from app.models.call_session import CallSession
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.calendar import BusinessHoursUpsert
from app.services.appointment_reservation_service import appointment_reservation_service
from app.services.calendar_service import calendar_service
from app.services.call_session_contact_state import (
    apply_post_call_recovery,
    get_contact_intake,
    sync_contact_intake_after_message,
)
from app.services.post_call_appointment_service import post_call_appointment_service
from app.utils.voice_contact_extraction import (
    _clean_email_stt_artifacts,
    strict_contact_email_from_text,
)


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(_type, compiler, **kw):
    return "JSON"


engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


TENANT_TZ = "Asia/Karachi"
# Dynamically find next Monday in the future to keep slots valid relative to date.today()
_today = date.today()
_days_ahead = 0 - _today.weekday()
if _days_ahead <= 0:
    _days_ahead += 7
MONDAY = _today + timedelta(days=_days_ahead)


@pytest.fixture()
def res_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    tenant = Tenant(name="ResTenant", schema_name="res_tenant")
    db.add(tenant)
    db.commit()
    u = User(
        first_name="T",
        last_name="U",
        email="t-%s@example.com" % uuid.uuid4().hex[:8],
        hashed_password="x",
        current_tenant_id=tenant.id,
    )
    db.add(u)
    db.commit()
    a = Agent(
        tenant_id=tenant.id,
        name="TestAgent",
        created_by=u.id,
        updated_by=u.id,
    )
    db.add(a)
    db.commit()
    calendar_service.create_business_hours(
        db,
        tenant.id,
        [
            BusinessHoursUpsert(
                day_of_week=MONDAY.weekday(),
                open_time="09:00",
                close_time="17:00",
                timezone=TENANT_TZ,
            )
        ],
    )
    try:
        yield db, tenant, u, a
    finally:
        db.close()


def _call_session(db, tenant, u, a) -> CallSession:
    cs = CallSession(
        user_id=u.id,
        agent_id=a.id,
        tenant_id=tenant.id,
        start_time=datetime.now(timezone.utc),
        call_type="inbound",
        status="active",
        from_number="+15550001111",
    )
    db.add(cs)
    db.commit()
    return cs


# --------------------------------------------------------------------------
# 1) Email STT-artifact cleanup
# --------------------------------------------------------------------------


def test_email_cleanup_strips_commas_inside_local_part():
    assert _clean_email_stt_artifacts("ali.sa,ee,b@gmail.com") == "ali.saeeb@gmail.com"


def test_email_cleanup_leaves_clean_input_unchanged():
    assert _clean_email_stt_artifacts("alex@gmail.com") == "alex@gmail.com"


def test_email_cleanup_handles_comma_separators_in_local_and_domain():
    # STT injected stray commas at multiple positions; cleanup recovers a valid form.
    assert (
        _clean_email_stt_artifacts("alex,carter@gmail.com")
        == "alexcarter@gmail.com"
    )


def test_email_cleanup_preserves_surrounding_text():
    out = _clean_email_stt_artifacts("My email is ali.sa,ee,b@gmail.com please")
    assert "ali.saeeb@gmail.com" in out
    assert out.startswith("My email is ")
    assert out.endswith(" please")


def test_email_cleanup_no_at_symbol_returns_input():
    assert _clean_email_stt_artifacts("hello world") == "hello world"


def test_strict_email_now_recovers_from_stt_commas():
    # Existing strict behavior (clean address) is preserved.
    assert strict_contact_email_from_text("reach me at a.b@gmail.com") == "a.b@gmail.com"
    # New behavior: STT-corrupted address is recovered via cleanup pass.
    assert (
        strict_contact_email_from_text("ali.sa,ee,b@gmail.com")
        == "ali.saeeb@gmail.com"
    )


def test_strict_email_cleanup_disabled_keeps_legacy_partial_match(monkeypatch):
    """
    When cleanup is OFF, we fall back to the historical literal regex which
    would lock onto the last valid-looking sub-span ("b@gmail.com") after the
    comma. With cleanup ON (default), the longer "ali.saeeb@gmail.com" wins.
    This test pins the documented disable behavior so it cannot drift silently.
    """
    monkeypatch.setattr(settings, "EMAIL_STT_CLEANUP_ENABLED", False)
    legacy = strict_contact_email_from_text("ali.sa,ee,b@gmail.com")
    assert legacy == "b@gmail.com"
    monkeypatch.setattr(settings, "EMAIL_STT_CLEANUP_ENABLED", True)
    recovered = strict_contact_email_from_text("ali.sa,ee,b@gmail.com")
    assert recovered == "ali.saeeb@gmail.com"


# --------------------------------------------------------------------------
# 2) Natural-confirmation path: agent repeats name + caller affirms
# --------------------------------------------------------------------------


def test_natural_confirmation_marks_name_confident(res_db):
    db, tenant, u, a = res_db
    cs = _call_session(db, tenant, u, a)
    from app.services.transcript_service import transcript_service

    transcript_service.add_message(
        db, cs.id, "agent",
        "Just to confirm, your name is Alex Carter, right?",
        "speech",
    )
    sync_contact_intake_after_message(
        db, cs.id, role="agent",
        message="Just to confirm, your name is Alex Carter, right?",
    )
    transcript_service.add_message(db, cs.id, "client", "Yes.", "speech")
    sync_contact_intake_after_message(db, cs.id, role="client", message="Yes.")

    db.refresh(cs)
    intake = get_contact_intake(cs)
    assert intake["name_confident"] is True
    assert intake["name"] == "Alex Carter"
    # Soft path must NOT claim the stronger "spelled" provenance.
    assert intake["name_spelled_confirmed"] is False


def test_natural_confirmation_does_not_fire_without_affirmation(res_db):
    db, tenant, u, a = res_db
    cs = _call_session(db, tenant, u, a)
    from app.services.transcript_service import transcript_service

    transcript_service.add_message(
        db, cs.id, "agent",
        "Just to confirm, your name is Alex Carter, right?",
        "speech",
    )
    sync_contact_intake_after_message(
        db, cs.id, role="agent",
        message="Just to confirm, your name is Alex Carter, right?",
    )
    transcript_service.add_message(db, cs.id, "client", "I want a checkup.", "speech")
    sync_contact_intake_after_message(
        db, cs.id, role="client", message="I want a checkup.",
    )

    db.refresh(cs)
    intake = get_contact_intake(cs)
    assert intake["name_confident"] is False
    assert intake["name"] is None


def test_natural_confirmation_skipped_when_awaiting_spell(res_db):
    db, tenant, u, a = res_db
    cs = _call_session(db, tenant, u, a)
    from app.services.transcript_service import transcript_service

    # Agent explicitly enters "spell your name" mode.
    transcript_service.add_message(
        db, cs.id, "agent", "Please spell your full name for me.", "speech",
    )
    sync_contact_intake_after_message(
        db, cs.id, role="agent", message="Please spell your full name for me.",
    )
    # Caller says "Yes" (which would otherwise be an affirmation, but spelling
    # context is active — must not fire natural confirmation).
    transcript_service.add_message(db, cs.id, "client", "Yes.", "speech")
    sync_contact_intake_after_message(db, cs.id, role="client", message="Yes.")

    db.refresh(cs)
    intake = get_contact_intake(cs)
    assert intake["name_confident"] is False


def test_natural_confirmation_disabled_via_flag(res_db, monkeypatch):
    monkeypatch.setattr(settings, "VOICE_NATURAL_NAME_CONFIRMATION", False)
    db, tenant, u, a = res_db
    cs = _call_session(db, tenant, u, a)
    from app.services.transcript_service import transcript_service

    transcript_service.add_message(
        db, cs.id, "agent",
        "Just to confirm, your name is Alex Carter, right?",
        "speech",
    )
    sync_contact_intake_after_message(
        db, cs.id, role="agent",
        message="Just to confirm, your name is Alex Carter, right?",
    )
    transcript_service.add_message(db, cs.id, "client", "Yes.", "speech")
    sync_contact_intake_after_message(db, cs.id, role="client", message="Yes.")

    db.refresh(cs)
    intake = get_contact_intake(cs)
    assert intake["name_confident"] is False
    monkeypatch.setattr(settings, "VOICE_NATURAL_NAME_CONFIRMATION", True)


def test_natural_confirmation_does_not_overwrite_spelled_name(res_db):
    db, tenant, u, a = res_db
    cs = _call_session(db, tenant, u, a)
    from app.services.transcript_service import transcript_service

    # First lock-in via spelling.
    transcript_service.add_message(
        db, cs.id, "agent", "Please spell your name.", "speech",
    )
    sync_contact_intake_after_message(
        db, cs.id, role="agent", message="Please spell your name.",
    )
    transcript_service.add_message(db, cs.id, "client", "J O H N", "speech")
    sync_contact_intake_after_message(db, cs.id, role="client", message="J O H N")

    # Then agent confirms a different name and caller says yes.
    transcript_service.add_message(
        db, cs.id, "agent",
        "Just to confirm, your name is Alex Carter.",
        "speech",
    )
    sync_contact_intake_after_message(
        db, cs.id, role="agent",
        message="Just to confirm, your name is Alex Carter.",
    )
    transcript_service.add_message(db, cs.id, "client", "Yes.", "speech")
    sync_contact_intake_after_message(db, cs.id, role="client", message="Yes.")

    db.refresh(cs)
    intake = get_contact_intake(cs)
    # Spelling provenance wins: confidence already True locks the field.
    assert intake["name_confident"] is True
    assert intake["name_spelled_confirmed"] is True
    assert intake["name"] == "John"


# --------------------------------------------------------------------------
# 3) apply_post_call_recovery — upgrade-only semantics
# --------------------------------------------------------------------------


def test_recovery_upgrades_unconfident_intake(res_db):
    db, tenant, u, a = res_db
    cs = _call_session(db, tenant, u, a)

    out = apply_post_call_recovery(
        db, cs,
        name="Alex Carter",
        email="alex@example.com",
        name_confident=True,
        email_confident=True,
    )
    db.refresh(cs)

    assert out["name_confident"] is True
    assert out["name"] == "Alex Carter"
    assert out["email_validated"] is True
    assert out["email"] == "alex@example.com"


def test_recovery_never_downgrades_existing_confidence(res_db):
    db, tenant, u, a = res_db
    cs = _call_session(db, tenant, u, a)
    cs.call_metadata = {
        "contact_intake": {
            "name": "John",
            "email": "john@example.com",
            "name_spelled_confirmed": True,
            "email_spelled_confirmed": True,
            "name_confident": True,
            "email_validated": True,
            "name_spell_failures": 0,
            "awaiting_spell_field": None,
        }
    }
    db.add(cs)
    db.commit()

    apply_post_call_recovery(
        db, cs,
        name="Different Person",
        email="other@example.com",
        name_confident=True,
        email_confident=True,
    )
    db.refresh(cs)
    intake = get_contact_intake(cs)
    # Existing confident intake is preserved verbatim.
    assert intake["name"] == "John"
    assert intake["email"] == "john@example.com"


def test_recovery_ignores_unconfident_inputs(res_db):
    db, tenant, u, a = res_db
    cs = _call_session(db, tenant, u, a)

    apply_post_call_recovery(
        db, cs,
        name="Alex Carter",
        email="alex@example.com",
        name_confident=False,
        email_confident=False,
    )
    db.refresh(cs)
    intake = get_contact_intake(cs)
    assert intake["name_confident"] is False
    assert intake["name"] is None
    assert intake["email_validated"] is False
    assert intake["email"] is None


# --------------------------------------------------------------------------
# 4) Post-call LLM recovery wiring in PostCallAppointmentService
# --------------------------------------------------------------------------


def _seed_unconfident_session_with_slot(db, tenant, u, a):
    day_slots = calendar_service.get_available_slots(db, tenant.id, MONDAY)
    slot = day_slots.slots[0].slot_start
    cs = _call_session(db, tenant, u, a)
    cs.call_metadata = {
        "contact_intake": {
            "name": None,
            "email": None,
            "name_spelled_confirmed": False,
            "email_spelled_confirmed": False,
            "name_confident": False,
            "email_validated": False,
            "name_spell_failures": 0,
            "awaiting_spell_field": None,
        },
        "booking_intent": {
            "slot_start_iso": slot.isoformat(),
            "appointment_reason": "checkup",
        },
    }
    db.add(cs)
    db.commit()

    from app.services.transcript_service import transcript_service
    transcript_service.add_message(db, cs.id, "agent", "Your full name?", "speech")
    transcript_service.add_message(
        db, cs.id, "client", "My full name is Alex Carter.", "speech",
    )
    return cs, slot


def test_post_call_llm_recovery_succeeds_and_books(res_db, monkeypatch):
    db, tenant, u, a = res_db
    monkeypatch.setattr(settings, "POST_CALL_LLM_CONTACT_RECOVERY", True)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")

    cs, _slot = _seed_unconfident_session_with_slot(db, tenant, u, a)

    fake_resp = {
        "content": (
            '{"name": "Alex Carter", "email": "alex@example.com", '
            '"name_confident": true, "email_confident": true}'
        )
    }
    with patch(
        "app.services.post_call_appointment_service.openai_service.chat_completion",
        return_value=fake_resp,
    ):
        post_call_appointment_service.process_call_session(db, cs.id)

    db.refresh(cs)
    assert cs.call_metadata.get("post_call_appointment") == "success"
    assert cs.call_metadata.get("post_call_contact_recovery") == "llm_succeeded"
    appts = db.query(Appointment).filter(Appointment.call_session_id == cs.id).all()
    assert len(appts) == 1
    assert appts[0].customer_name == "Alex Carter"


def test_post_call_llm_recovery_disabled_keeps_existing_failure(res_db, monkeypatch):
    db, tenant, u, a = res_db
    monkeypatch.setattr(settings, "POST_CALL_LLM_CONTACT_RECOVERY", False)

    cs, _slot = _seed_unconfident_session_with_slot(db, tenant, u, a)

    post_call_appointment_service.process_call_session(db, cs.id)

    db.refresh(cs)
    assert cs.call_metadata.get("post_call_appointment") == "failed"
    assert cs.call_metadata.get("post_call_appointment_detail") == "contact_not_confident"
    assert cs.call_metadata.get("post_call_contact_recovery") is None
    assert db.query(Appointment).filter(Appointment.call_session_id == cs.id).count() == 0


def test_post_call_llm_recovery_rejects_assistant_name(res_db, monkeypatch):
    db, tenant, u, a = res_db
    monkeypatch.setattr(settings, "POST_CALL_LLM_CONTACT_RECOVERY", True)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")

    cs, _slot = _seed_unconfident_session_with_slot(db, tenant, u, a)

    # Defensive: if the LLM hallucinates "AI Assistant" as the name, we drop it.
    fake_resp = {
        "content": (
            '{"name": "The Assistant", "email": null, '
            '"name_confident": true, "email_confident": false}'
        )
    }
    with patch(
        "app.services.post_call_appointment_service.openai_service.chat_completion",
        return_value=fake_resp,
    ):
        post_call_appointment_service.process_call_session(db, cs.id)

    db.refresh(cs)
    assert cs.call_metadata.get("post_call_appointment") == "failed"
    assert cs.call_metadata.get("post_call_appointment_detail") == "contact_not_confident"


def test_post_call_llm_recovery_validates_email_locally(res_db, monkeypatch):
    db, tenant, u, a = res_db
    monkeypatch.setattr(settings, "POST_CALL_LLM_CONTACT_RECOVERY", True)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")

    cs, _slot = _seed_unconfident_session_with_slot(db, tenant, u, a)

    # LLM provides bogus email — must be dropped, name still wins for booking.
    fake_resp = {
        "content": (
            '{"name": "Alex Carter", "email": "not-an-email", '
            '"name_confident": true, "email_confident": true}'
        )
    }
    with patch(
        "app.services.post_call_appointment_service.openai_service.chat_completion",
        return_value=fake_resp,
    ):
        post_call_appointment_service.process_call_session(db, cs.id)

    db.refresh(cs)
    assert cs.call_metadata.get("post_call_appointment") == "success"
    appts = db.query(Appointment).filter(Appointment.call_session_id == cs.id).all()
    assert len(appts) == 1
    assert appts[0].customer_name == "Alex Carter"
    # Bogus email was rejected at validation time.
    assert (appts[0].customer_email or "") == ""


def test_post_call_llm_recovery_handles_openai_exception(res_db, monkeypatch):
    db, tenant, u, a = res_db
    monkeypatch.setattr(settings, "POST_CALL_LLM_CONTACT_RECOVERY", True)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")

    cs, _slot = _seed_unconfident_session_with_slot(db, tenant, u, a)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("openai down")

    with patch(
        "app.services.post_call_appointment_service.openai_service.chat_completion",
        side_effect=_raise,
    ):
        post_call_appointment_service.process_call_session(db, cs.id)

    db.refresh(cs)
    # Graceful degradation: original failure path still triggers.
    assert cs.call_metadata.get("post_call_appointment") == "failed"
    assert cs.call_metadata.get("post_call_appointment_detail") == "contact_not_confident"


def test_post_call_llm_recovery_skipped_when_intake_already_confident(res_db, monkeypatch):
    """Contact-recovery LLM must NOT run when name + validated email already present."""
    db, tenant, u, a = res_db
    monkeypatch.setattr(settings, "POST_CALL_LLM_CONTACT_RECOVERY", True)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")

    day_slots = calendar_service.get_available_slots(db, tenant.id, MONDAY)
    slot = day_slots.slots[0].slot_start
    cs = _call_session(db, tenant, u, a)
    cs.call_metadata = {
        "contact_intake": {
            "name": "John",
            "email": "john@example.com",
            "name_spelled_confirmed": True,
            "email_spelled_confirmed": True,
            "name_confident": True,
            "email_validated": True,
            "name_spell_failures": 0,
            "awaiting_spell_field": None,
        },
        "booking_intent": {"slot_start_iso": slot.isoformat()},
    }
    db.add(cs)
    db.commit()

    from app.services.transcript_service import transcript_service
    transcript_service.add_message(db, cs.id, "client", "Hi.", "speech")

    with patch(
        "app.services.post_call_appointment_service.openai_service.chat_completion",
    ) as mock_llm:
        post_call_appointment_service.process_call_session(db, cs.id)
        # Non-PII LLM extraction may still run; contact-recovery LLM must not.
        for call in mock_llm.call_args_list:
            kwargs = call.kwargs or {}
            sysp = (kwargs.get("system_prompt") or "")
            assert "caller's contact details" not in sysp.lower()

    db.refresh(cs)
    assert cs.call_metadata.get("post_call_appointment") == "success"
    assert cs.call_metadata.get("post_call_contact_recovery") is None


def test_post_call_llm_recovery_fills_email_when_name_already_confident(res_db, monkeypatch):
    """If name is confident but email missing, contact LLM still runs and can set email."""
    db, tenant, u, a = res_db
    monkeypatch.setattr(settings, "POST_CALL_LLM_CONTACT_RECOVERY", True)
    monkeypatch.setattr(settings, "POST_CALL_LLM_EMAIL_RECOVERY_WHEN_NAME_OK", True)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")

    day_slots = calendar_service.get_available_slots(db, tenant.id, MONDAY)
    slot = day_slots.slots[0].slot_start
    cs = _call_session(db, tenant, u, a)
    cs.call_metadata = {
        "contact_intake": {
            "name": "Alex Carter",
            "email": None,
            "name_spelled_confirmed": False,
            "email_spelled_confirmed": False,
            "name_confident": True,
            "email_validated": False,
            "name_spell_failures": 0,
            "awaiting_spell_field": None,
        },
        "booking_intent": {"slot_start_iso": slot.isoformat(), "appointment_reason": "checkup"},
    }
    db.add(cs)
    db.commit()

    from app.services.transcript_service import transcript_service
    transcript_service.add_message(db, cs.id, "agent", "What is your email?", "speech")
    transcript_service.add_message(
        db, cs.id, "client", "My email is michaeljackson@therategmail.com.", "speech"
    )

    fake_resp = {
        "content": (
            '{"name": null, "email": "michaeljackson@therategmail.com", '
            '"name_confident": false, "email_confident": false}'
        )
    }
    with patch(
        "app.services.post_call_appointment_service.openai_service.chat_completion",
        return_value=fake_resp,
    ):
        post_call_appointment_service.process_call_session(db, cs.id)

    db.refresh(cs)
    intake = get_contact_intake(cs)
    assert intake.get("email_validated") is True
    assert intake.get("email") == "michaeljackson@therategmail.com"
    assert cs.call_metadata.get("post_call_contact_recovery") == "llm_succeeded"
    appts = db.query(Appointment).filter(Appointment.call_session_id == cs.id).all()
    assert len(appts) == 1
    assert appts[0].customer_email == "michaeljackson@therategmail.com"


def test_recover_contact_llm_anchor_trust_off_keeps_conservative_email(res_db, monkeypatch):
    monkeypatch.setattr(settings, "POST_CALL_LLM_CONTACT_RECOVERY", True)
    monkeypatch.setattr(settings, "POST_CALL_LLM_EMAIL_ANCHOR_TRUST", False)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    tr = (
        "AGENT: Thank you — can you provide your email address please?\n"
        "CLIENT: michaeljackson@therategmail.com.\n"
    )
    fake_resp = {
        "content": (
            '{"name": null, "email": "michaeljackson@therategmail.com", '
            '"name_confident": false, "email_confident": false}'
        )
    }
    with patch(
        "app.services.post_call_appointment_service.openai_service.chat_completion",
        return_value=fake_resp,
    ):
        out = post_call_appointment_service._recover_contact_via_llm(tr)
    assert out.get("email") == "michaeljackson@therategmail.com"
    assert out.get("email_confident") is False


def test_recover_contact_llm_anchor_trust_on_promotes_email(res_db, monkeypatch):
    monkeypatch.setattr(settings, "POST_CALL_LLM_CONTACT_RECOVERY", True)
    monkeypatch.setattr(settings, "POST_CALL_LLM_EMAIL_ANCHOR_TRUST", True)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    tr = (
        "AGENT: Thank you — can you provide your email address please?\n"
        "CLIENT: michaeljackson@therategmail.com.\n"
    )
    fake_resp = {
        "content": (
            '{"name": null, "email": "michaeljackson@therategmail.com", '
            '"name_confident": false, "email_confident": false}'
        )
    }
    with patch(
        "app.services.post_call_appointment_service.openai_service.chat_completion",
        return_value=fake_resp,
    ):
        out = post_call_appointment_service._recover_contact_via_llm(tr)
    assert out.get("email") == "michaeljackson@therategmail.com"
    assert out.get("email_confident") is True
