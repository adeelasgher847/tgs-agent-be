"""Unit tests for the deterministic Gemini prompt sanitizer."""
import pytest

from app.utils.gemini_prompt_sanitizer import sanitize_prompt_for_gemini


def test_empty_string_returns_empty():
    assert sanitize_prompt_for_gemini("") == ""


def test_whitespace_only_returns_empty():
    assert sanitize_prompt_for_gemini("   \n\t  ") == ""


def test_plain_prompt_gets_role_prefix():
    result = sanitize_prompt_for_gemini("Answer customer questions politely.")
    assert result.startswith("You are an AI voice agent for phone calls.\n\n")
    assert "Answer customer questions politely." in result


def test_prompt_starting_with_you_are_no_duplicate_prefix():
    prompt = "You are a helpful sales agent. Be concise."
    result = sanitize_prompt_for_gemini(prompt)
    # Should not prepend the default prefix
    assert result.count("You are") == 1
    assert "helpful sales agent" in result


def test_prompt_starting_with_youre_no_prefix():
    prompt = "You're a scheduling assistant. Book appointments."
    result = sanitize_prompt_for_gemini(prompt)
    assert result.count("You") == 1
    assert not result.startswith("You are an AI")


def test_markdown_fences_stripped():
    prompt = "```\nYou are a bot.\nBe helpful.\n```"
    result = sanitize_prompt_for_gemini(prompt)
    assert "```" not in result
    assert "You are a bot." in result


def test_markdown_fences_with_language_tag_stripped():
    prompt = "```python\nYou are a bot.\n```"
    result = sanitize_prompt_for_gemini(prompt)
    assert "```" not in result
    assert "python" not in result


def test_windows_line_endings_normalized():
    prompt = "You are an agent.\r\nBe helpful.\r\nThank you."
    result = sanitize_prompt_for_gemini(prompt)
    assert "\r" not in result


def test_excess_blank_lines_collapsed():
    prompt = "You are an agent.\n\n\n\n\nBe helpful."
    result = sanitize_prompt_for_gemini(prompt)
    # At most two consecutive newlines
    assert "\n\n\n" not in result


def test_trailing_whitespace_per_line_stripped():
    prompt = "You are an agent.   \nBe helpful.   "
    result = sanitize_prompt_for_gemini(prompt)
    for line in result.split("\n"):
        assert line == line.rstrip()


def test_single_trailing_newline():
    prompt = "You are an agent.\n\n\n"
    result = sanitize_prompt_for_gemini(prompt)
    assert result.endswith("\n")
    assert not result.endswith("\n\n")


def test_null_bytes_removed():
    prompt = "You are an agent.\x00 Be helpful.\x00"
    result = sanitize_prompt_for_gemini(prompt)
    assert "\x00" not in result
    assert "Be helpful." in result


def test_control_chars_removed_but_tabs_kept():
    prompt = "You are an agent.\tBe concise.\x01\x07"
    result = sanitize_prompt_for_gemini(prompt)
    assert "\x01" not in result
    assert "\x07" not in result
    assert "\t" in result  # tabs are preserved
