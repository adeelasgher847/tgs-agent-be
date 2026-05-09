"""Deterministic voice contact extraction."""
from app.utils.voice_contact_extraction import (
    client_lines_from_transcript_text,
    extract_contact_from_client_lines,
    extract_spelled_name_from_line,
    strict_contact_email_from_text,
)


def test_strict_email_accepts_valid():
    assert strict_contact_email_from_text("reach me at a.b@gmail.com thanks") == "a.b@gmail.com"


def test_strict_email_rejects_no_dot_domain():
    assert strict_contact_email_from_text("a@b") is None


def test_extract_spelled_name():
    assert extract_spelled_name_from_line("J O H N") == "John"


def test_extract_contact_newest_first_prefers_recent():
    lines = [
        "J O H N",
        "noise",
    ]
    out = extract_contact_from_client_lines(lines)
    assert out["name"] == "John"


def test_client_lines_from_transcript_text_order():
    text = "CLIENT: old\nCLIENT: new"
    assert client_lines_from_transcript_text(text) == ["new", "old"]
