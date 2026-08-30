"""Deterministic voice contact extraction."""
from app.utils.voice_contact_extraction import (
    client_lines_from_transcript_text,
    extract_contact_from_client_lines,
    extract_spelled_name_from_line,
    is_caller_id_phone_reference,
    plausible_address_from_text,
    strict_contact_email_from_text,
    strict_contact_phone_from_text,
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


def test_strict_phone_accepts_digit_run():
    assert strict_contact_phone_from_text("it's 5551234567") == "555-123-4567"


def test_strict_phone_accepts_formatted_number():
    assert strict_contact_phone_from_text("call me at 555-123-4567") == "555-123-4567"


def test_strict_phone_strips_leading_country_code_one():
    assert strict_contact_phone_from_text("15551234567") == "555-123-4567"


def test_strict_phone_accepts_spoken_digit_words():
    assert (
        strict_contact_phone_from_text("five five five one two three four five six seven")
        == "555-123-4567"
    )


def test_strict_phone_rejects_short_number():
    assert strict_contact_phone_from_text("12345") is None


def test_strict_phone_rejects_non_numeric_garbled_line():
    # Real-world garbled transcript case: no usable digits at all.
    assert (
        strict_contact_phone_from_text(
            "Yeah. The the number we call mister I'm calling you."
        )
        is None
    )


def test_is_caller_id_phone_reference_matches_common_phrasing():
    assert is_caller_id_phone_reference("just use the number you called me from")
    assert is_caller_id_phone_reference("this number is fine")
    assert is_caller_id_phone_reference("same number please")


def test_is_caller_id_phone_reference_false_for_unrelated_text():
    assert not is_caller_id_phone_reference("my number is 555-123-4567")


def test_plausible_address_accepts_street_style_line():
    assert plausible_address_from_text("123 Main Street, Springfield") == (
        "123 Main Street, Springfield"
    )


def test_plausible_address_rejects_short_reply():
    assert plausible_address_from_text("yes") is None


def test_plausible_address_rejects_digits_only():
    assert plausible_address_from_text("123456789") is None


def test_plausible_address_rejects_words_only():
    assert plausible_address_from_text("somewhere downtown nearby") is None


def test_plausible_address_rejects_unrelated_speech_with_stray_digit():
    # Regression: a bare digit anywhere used to be enough to "look like" an
    # address, which silently locked in unrelated speech as a confirmed
    # service address (worse than the original re-ask loop it was meant to
    # avoid). Now requires either a street/unit keyword or the digit to lead.
    assert plausible_address_from_text(
        "It has been about 3 years since the last inspection"
    ) is None
    assert plausible_address_from_text(
        "I have 2 addresses actually, home and office"
    ) is None


def test_plausible_address_accepts_apartment_style_without_leading_digit():
    assert (
        plausible_address_from_text("Apartment 6B, Building 58, Sector B, New York")
        == "Apartment 6B, Building 58, Sector B, New York"
    )


def test_strict_phone_rejects_digits_scattered_across_self_correction():
    # Regression: digits were concatenated across the entire utterance, so a
    # caller's self-correction ("wait no") could get silently merged into a
    # fabricated 10-digit number instead of being rejected.
    assert (
        strict_contact_phone_from_text(
            "Sorry, hold on, its five five five one two three, wait no four five six seven"
        )
        is None
    )
