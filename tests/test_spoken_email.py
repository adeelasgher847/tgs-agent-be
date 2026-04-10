"""Unit tests for voice / STT email normalization."""
from app.utils.spoken_email import (
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


def test_resolve_prefers_valid_token_over_transcript():
    token = resolve_customer_email_for_booking(
        token_email_raw="other@company.org",
        transcript_client_lines_newest_first=["reach me at foo@gmail.com"],
    )
    assert token == "other@company.org"


def test_resolve_transcript_when_token_invalid():
    hit = resolve_customer_email_for_booking(
        token_email_raw="not-an-email",
        transcript_client_lines_newest_first=["backup is jane.doe@site.co.uk"],
    )
    assert hit == "jane.doe@site.co.uk"


def test_normalize_stored_email_valid():
    n = normalize_stored_email("  user@example.com ")
    assert n is not None
    assert n.endswith("@example.com")


def test_normalize_stored_email_invalid():
    assert normalize_stored_email("not-an-email") is None
    assert normalize_stored_email(None) is None
    assert normalize_stored_email("") is None


def test_resolve_token_placeholder_skipped():
    assert (
        resolve_customer_email_for_booking(
            token_email_raw="none",
            transcript_client_lines_newest_first=["x@test.com"],
        )
        == "x@test.com"
    )
