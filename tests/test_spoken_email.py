"""Unit tests for voice / STT email normalization."""
from app.utils.spoken_email import (
    build_email_repair_prompt,
    best_email_from_client_utterances,
    coerce_email_from_text,
    normalize_stored_email,
    resolve_customer_email_for_booking,
)


def test_coerce_literal_email():
    assert coerce_email_from_text("reach me at foo.bar@gmail.com thanks") == "foo.bar@gmail.com"


def test_coerce_spoken_email():
    assert (
        coerce_email_from_text("my email is john dot smith at gmail dot com")
        == "john.smith@gmail.com"
    )


def test_coerce_rejects_garbage():
    assert coerce_email_from_text("no email here today") is None


def test_best_prefers_newest():
    newest_first = [
        "yes book it",
        "john dot smith at example dot com",
    ]
    assert best_email_from_client_utterances(newest_first) == "john.smith@example.com"


def test_resolve_prefers_transcript_over_valid_token():
    resolution = resolve_customer_email_for_booking(
        token_email_raw="other@company.org",
        transcript_client_lines_newest_first=["reach me at foo@gmail.com"],
    )
    assert resolution.verified_email is None
    assert resolution.pending_email == "foo@gmail.com"
    assert resolution.transcript_email == "foo@gmail.com"
    assert resolution.token_email == "other@company.org"
    assert resolution.suspicious_token_email is True
    assert resolution.source == "transcript_reconstructed"


def test_resolve_transcript_when_token_invalid():
    resolution = resolve_customer_email_for_booking(
        token_email_raw="not-an-email",
        transcript_client_lines_newest_first=["backup is jane.doe@site.co.uk"],
    )
    assert resolution.pending_email == "jane.doe@site.co.uk"
    assert resolution.verified_email is None
    assert resolution.source == "transcript_reconstructed"


def test_normalize_stored_email_valid():
    n = normalize_stored_email("  user@example.com ")
    assert n is not None
    assert n.endswith("@example.com")


def test_normalize_stored_email_invalid():
    assert normalize_stored_email("not-an-email") is None
    assert normalize_stored_email(None) is None
    assert normalize_stored_email("") is None


def test_resolve_token_placeholder_skipped():
    resolution = resolve_customer_email_for_booking(
        token_email_raw="none",
        transcript_client_lines_newest_first=["x@test.com"],
    )
    assert resolution.pending_email == "x@test.com"
    assert resolution.token_email is None


def test_resolve_marks_token_only_email_pending_or_untrusted():
    resolution = resolve_customer_email_for_booking(
        token_email_raw="ali.saidicp@gmail.com",
        transcript_client_lines_newest_first=[],
    )
    assert resolution.verified_email is None
    assert resolution.pending_email is None
    assert resolution.suspicious_token_email is True
    assert resolution.should_attempt_llm_repair is True
    assert resolution.source == "token_only_unverified"


def test_resolve_regression_prefers_transcript_for_fused_stt_email():
    resolution = resolve_customer_email_for_booking(
        token_email_raw="ali.saidicp@gmail.com",
        transcript_client_lines_newest_first=[
            "my email is ali dot saeed ict at gmail dot com",
        ],
    )
    assert resolution.verified_email is None
    assert resolution.pending_email == "ali.saeedict@gmail.com"
    assert resolution.suspicious_token_email is True
    assert resolution.source == "transcript_reconstructed"


def test_resolve_repeated_transcript_email_is_verified():
    resolution = resolve_customer_email_for_booking(
        token_email_raw=None,
        transcript_client_lines_newest_first=[
            "yes that's correct",
            "my email is john dot smith at gmail dot com",
            "john dot smith at gmail dot com",
        ],
    )
    assert resolution.verified_email == "john.smith@gmail.com"
    assert resolution.pending_email is None
    assert resolution.source == "explicit_user_confirmed"


def test_build_email_repair_prompt_mentions_transcript_priority():
    prompt = build_email_repair_prompt(
        token_email_raw="ali.saidicp@gmail.com",
        transcript_client_lines_newest_first=["ali dot saeed ict at gmail dot com"],
    )
    assert "Do not trust fused raw STT emails blindly" in prompt
    assert "ali dot saeed ict at gmail dot com" in prompt
