"""
Unit tests for app.services.call_session_contact_state.

Coverage:
  - default_contact_intake() includes the new phone/address fields with
    correct defaults (regression guard: name/email defaults unchanged).
  - apply_transcript_turn(): phone/address immediate-confirm-on-plausibility
    semantics (no echo-and-wait-for-no-correction round trip, unlike name).
  - Caller-id phone reference resolves against call_session.from_number.
  - Hard retry ceiling (MAX_FIELD_COLLECTION_FAILURES) stops the loop and
    accepts a best-effort value for phone/address.
  - build_contact_intake_prompt_block(): only lists confirmed fields, empty
    string when nothing confirmed.
  - Existing name/email behavior (spelling, self-intro, natural
    confirmation) is unchanged by the refactor.

A lightweight fake call_session (plain object with `.id`/`.call_metadata`)
and a MagicMock db are used rather than a real ORM row + sqlite session —
apply_transcript_turn/_save_contact_intake only ever read/write
`call_session.call_metadata` and call `db.add`/`db.commit`/`db.refresh`,
all no-ops on a Mock, matching this repo's own MagicMock-handler convention
for prompt/contact-intake unit tests (see tests/voice/test_bracket_tag_prompt_consistency.py).
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from app.services.call_session_contact_state import (
    MAX_FIELD_COLLECTION_FAILURES,
    MAX_NAME_SPELL_FAILURES,
    apply_transcript_turn,
    build_contact_intake_prompt_block,
    default_contact_intake,
    get_contact_intake,
)


class _FakeCallSession:
    def __init__(self, **kwargs):
        self.id = uuid.uuid4()
        self.call_metadata = {}
        self.from_number = None
        self.customer_phone_number = None
        for k, v in kwargs.items():
            setattr(self, k, v)


def _db() -> MagicMock:
    return MagicMock()


def test_max_field_collection_failures_preserves_effective_threshold():
    # Renaming MAX_NAME_SPELL_FAILURES -> MAX_FIELD_COLLECTION_FAILURES must not
    # change the effective retry threshold for existing name/email behavior.
    assert MAX_FIELD_COLLECTION_FAILURES == 3
    assert MAX_NAME_SPELL_FAILURES == 3


def test_default_contact_intake_has_new_phone_address_fields():
    intake = default_contact_intake()
    assert intake["phone"] is None
    assert intake["phone_confirmed"] is False
    assert intake["phone_collection_failures"] == 0
    assert intake["address"] is None
    assert intake["address_confirmed"] is False
    assert intake["address_collection_failures"] == 0
    # Unchanged existing fields.
    assert intake["name"] is None
    assert intake["email"] is None
    assert intake["name_spell_failures"] == 0


# ---- Phone -----------------------------------------------------------------


def test_phone_confirmed_immediately_on_valid_number():
    cs = _FakeCallSession()
    db = _db()
    apply_transcript_turn(
        db, cs, role="agent", message="What's the best phone number to reach you?",
        preceding_agent_text=None,
    )
    apply_transcript_turn(
        db, cs, role="client", message="It's 555-123-4567",
        preceding_agent_text="What's the best phone number to reach you?",
    )
    intake = get_contact_intake(cs)
    assert intake["phone"] == "555-123-4567"
    assert intake["phone_confirmed"] is True
    assert intake["phone_collection_failures"] == 0


def test_phone_caller_id_reference_resolves_from_call_session():
    cs = _FakeCallSession(from_number="+15551234567")
    db = _db()
    apply_transcript_turn(
        db, cs, role="agent", message="What's a good callback number?",
        preceding_agent_text=None,
    )
    apply_transcript_turn(
        db, cs, role="client", message="just use the number you called me from",
        preceding_agent_text="What's a good callback number?",
    )
    intake = get_contact_intake(cs)
    assert intake["phone"] == "+15551234567"
    assert intake["phone_confirmed"] is True


def test_phone_caller_id_reference_without_caller_id_counts_as_failure():
    cs = _FakeCallSession()  # no from_number/customer_phone_number on file
    db = _db()
    apply_transcript_turn(
        db, cs, role="agent", message="What's a good callback number?",
        preceding_agent_text=None,
    )
    apply_transcript_turn(
        db, cs, role="client",
        message="Yeah. The the number we call mister I'm calling you.",
        preceding_agent_text="What's a good callback number?",
    )
    intake = get_contact_intake(cs)
    assert intake["phone_confirmed"] is False
    assert intake["phone_collection_failures"] == 1


def test_phone_hard_retry_ceiling_accepts_best_effort():
    cs = _FakeCallSession()
    db = _db()
    for _ in range(MAX_FIELD_COLLECTION_FAILURES):
        apply_transcript_turn(
            db, cs, role="agent", message="What's your phone number?",
            preceding_agent_text=None,
        )
        apply_transcript_turn(
            db, cs, role="client", message="mumble garble nothing useful",
            preceding_agent_text="What's your phone number?",
        )
    intake = get_contact_intake(cs)
    # Ceiling reached: stop re-asking, accept best-effort rather than loop forever.
    assert intake["phone_collection_failures"] == MAX_FIELD_COLLECTION_FAILURES
    assert intake["phone_confirmed"] is True
    assert intake["phone"]


def test_phone_retry_ceiling_reached_even_when_agent_rephrases_each_retry():
    # Regression: an LLM-generated retry doesn't always match _ASK_PHONE_AGENT's
    # fixed phrasing ("phone number", "callback number", ...). If
    # awaiting_spell_field were cleared unconditionally every turn (as it used
    # to be), a rephrased retry like "Sorry, could you say that again?" would
    # silently drop out of phone-collection context and the failure counter
    # would stop incrementing — the exact structural gap this fix targets.
    cs = _FakeCallSession()
    db = _db()
    rephrasings = [
        "What's your phone number?",
        "Sorry, could you say that again?",
        "One more time, please?",
    ]
    for agent_text in rephrasings:
        apply_transcript_turn(
            db, cs, role="agent", message=agent_text, preceding_agent_text=None,
        )
        apply_transcript_turn(
            db, cs, role="client", message="mumble garble nothing useful",
            preceding_agent_text=agent_text,
        )
    intake = get_contact_intake(cs)
    assert intake["phone_collection_failures"] == MAX_FIELD_COLLECTION_FAILURES
    assert intake["phone_confirmed"] is True


# ---- Address -----------------------------------------------------------------


def test_address_confirmed_immediately_on_plausible_text():
    cs = _FakeCallSession()
    db = _db()
    apply_transcript_turn(
        db, cs, role="agent", message="What's your service address?",
        preceding_agent_text=None,
    )
    apply_transcript_turn(
        db, cs, role="client", message="123 Main Street, Springfield",
        preceding_agent_text="What's your service address?",
    )
    intake = get_contact_intake(cs)
    assert intake["address"] == "123 Main Street, Springfield"
    assert intake["address_confirmed"] is True
    assert intake["address_collection_failures"] == 0


def test_address_short_reply_counts_as_failure_not_confirmed():
    cs = _FakeCallSession()
    db = _db()
    apply_transcript_turn(
        db, cs, role="agent", message="Where are you located?",
        preceding_agent_text=None,
    )
    apply_transcript_turn(
        db, cs, role="client", message="uh yeah",
        preceding_agent_text="Where are you located?",
    )
    intake = get_contact_intake(cs)
    assert intake["address_confirmed"] is False
    assert intake["address_collection_failures"] == 1


def test_address_hard_retry_ceiling_accepts_best_effort():
    cs = _FakeCallSession()
    db = _db()
    for _ in range(MAX_FIELD_COLLECTION_FAILURES):
        apply_transcript_turn(
            db, cs, role="agent", message="What's your address?",
            preceding_agent_text=None,
        )
        apply_transcript_turn(
            db, cs, role="client", message="not sure honestly",
            preceding_agent_text="What's your address?",
        )
    intake = get_contact_intake(cs)
    assert intake["address_collection_failures"] == MAX_FIELD_COLLECTION_FAILURES
    assert intake["address_confirmed"] is True
    assert intake["address"]


# ---- Prompt injection block -------------------------------------------------


def test_prompt_block_empty_when_nothing_confirmed():
    intake = default_contact_intake()
    assert build_contact_intake_prompt_block(intake) == ""


def test_prompt_block_lists_only_confirmed_fields():
    intake = default_contact_intake()
    intake["name"] = "Adel"
    intake["name_confident"] = True
    intake["phone"] = "555-123-4567"
    intake["phone_confirmed"] = True
    # email/address left unconfirmed.
    block = build_contact_intake_prompt_block(intake)
    assert "Name: CONFIRMED (Adel)" in block
    assert "Phone: CONFIRMED (555-123-4567)" in block
    assert "Email" not in block
    assert "Address" not in block


def test_prompt_block_includes_all_four_fields_when_confirmed():
    intake = default_contact_intake()
    intake.update(
        name="Adel",
        name_confident=True,
        email="adel@example.com",
        email_validated=True,
        phone="555-123-4567",
        phone_confirmed=True,
        address="123 Main Street",
        address_confirmed=True,
    )
    block = build_contact_intake_prompt_block(intake)
    assert "Name: CONFIRMED (Adel)" in block
    assert "Email: CONFIRMED (adel@example.com)" in block
    assert "Phone: CONFIRMED (555-123-4567)" in block
    assert "Address: CONFIRMED (123 Main Street)" in block
    assert "DO NOT RE-ASK" in block


# ---- Regression: existing name/email behavior unchanged --------------------


def test_name_spelling_flow_unchanged():
    cs = _FakeCallSession()
    db = _db()
    apply_transcript_turn(
        db, cs, role="agent", message="Can you spell your name for me?",
        preceding_agent_text=None,
    )
    apply_transcript_turn(
        db, cs, role="client", message="J O H N",
        preceding_agent_text="Can you spell your name for me?",
    )
    intake = get_contact_intake(cs)
    assert intake["name"] == "John"
    assert intake["name_spelled_confirmed"] is True
    assert intake["name_confident"] is True


def test_email_spelling_flow_unchanged():
    cs = _FakeCallSession()
    db = _db()
    apply_transcript_turn(
        db, cs, role="agent", message="Could you spell your email address?",
        preceding_agent_text=None,
    )
    apply_transcript_turn(
        db, cs, role="client", message="john@example.com",
        preceding_agent_text="Could you spell your email address?",
    )
    intake = get_contact_intake(cs)
    assert intake["email"] == "john@example.com"
    assert intake["email_validated"] is True


def test_name_self_introduction_flow_unchanged():
    cs = _FakeCallSession()
    db = _db()
    apply_transcript_turn(
        db, cs, role="client", message="Hi, my name is Nishan",
        preceding_agent_text=None,
    )
    intake = get_contact_intake(cs)
    assert intake["name"] == "Nishan"
    assert intake["name_self_introduced"] is True
    assert intake["name_confident"] is False  # awaits agent echo

    apply_transcript_turn(
        db, cs, role="agent", message="Great, Nishan, how can I help?",
        preceding_agent_text=None,
    )
    intake = get_contact_intake(cs)
    assert intake["name_confident"] is True
