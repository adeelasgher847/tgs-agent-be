"""
Regression coverage for _model_supports_thinking's exact-match-or-versioned-
suffix semantics. A plain startswith() check would incorrectly match
"gemini-2.5-flash-lite" against the allowlisted "gemini-2.5-flash" entry,
even though it's a distinct variant that doesn't support thinking_config.
"""
from app.services.gemini_service import _model_supports_thinking


def test_exact_allowlisted_models_supported():
    assert _model_supports_thinking("gemini-2.5-pro") is True
    assert _model_supports_thinking("gemini-2.5-flash") is True
    assert _model_supports_thinking("gemini-2.5-pro-preview") is True


def test_versioned_suffix_still_supported():
    assert _model_supports_thinking("gemini-2.5-flash-001") is True
    assert _model_supports_thinking("gemini-2.5-pro-20250115") is True


def test_lite_variant_not_supported_despite_shared_prefix():
    assert _model_supports_thinking("gemini-2.5-flash-lite") is False


def test_unrelated_model_not_supported():
    assert _model_supports_thinking("gemini-1.5-flash") is False
    assert _model_supports_thinking("gpt-4o-mini") is False
