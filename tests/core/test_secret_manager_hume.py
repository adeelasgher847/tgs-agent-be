"""
Unit tests for ``app.core.secret_manager.get_hume_api_key`` — mirrors the
existing (only implicitly-covered-via-RimeTTSAdapter) contract for
``get_rime_api_key``: development reads a plain env value or raises
``ValueError``; staging/production try GCP Secret Manager first, fall back
to the env var, and raise ``RuntimeError`` if neither is available.

No real GCP credentials or network calls — ``_fetch_from_secret_manager`` is
patched at the module boundary.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.core.secret_manager import get_hume_api_key


@pytest.fixture(autouse=True)
def _clear_cache():
    """get_hume_api_key is @lru_cache'd — clear before/after every test so
    tests don't leak state into each other."""
    get_hume_api_key.cache_clear()
    yield
    get_hume_api_key.cache_clear()


class TestGetHumeApiKeyDevelopment:
    def test_returns_env_value_in_development(self):
        with patch("app.core.secret_manager.settings") as mock_settings:
            mock_settings.ENVIRONMENT = "development"
            mock_settings.HUME_API_KEY = "dev-hume-key"
            assert get_hume_api_key() == "dev-hume-key"

    def test_strips_whitespace_in_development(self):
        with patch("app.core.secret_manager.settings") as mock_settings:
            mock_settings.ENVIRONMENT = "development"
            mock_settings.HUME_API_KEY = "  dev-hume-key  "
            assert get_hume_api_key() == "dev-hume-key"

    def test_raises_value_error_when_unset_in_development(self):
        with patch("app.core.secret_manager.settings") as mock_settings:
            mock_settings.ENVIRONMENT = "development"
            mock_settings.HUME_API_KEY = ""
            with pytest.raises(ValueError, match="HUME_API_KEY is not set"):
                get_hume_api_key()

    def test_raises_value_error_when_only_whitespace_in_development(self):
        with patch("app.core.secret_manager.settings") as mock_settings:
            mock_settings.ENVIRONMENT = "development"
            mock_settings.HUME_API_KEY = "   "
            with pytest.raises(ValueError, match="HUME_API_KEY is not set"):
                get_hume_api_key()


class TestGetHumeApiKeyStagingProduction:
    @pytest.mark.parametrize("env", ["staging", "production"])
    def test_prefers_secret_manager_value(self, env):
        with patch("app.core.secret_manager.settings") as mock_settings, patch(
            "app.core.secret_manager._fetch_from_secret_manager",
            return_value="secret-manager-hume-key",
        ) as mock_fetch:
            mock_settings.ENVIRONMENT = env
            mock_settings.HUME_API_KEY = "env-fallback-should-not-be-used"
            assert get_hume_api_key() == "secret-manager-hume-key"
            mock_fetch.assert_called_once_with("HUME_API_KEY")

    @pytest.mark.parametrize("env", ["staging", "production"])
    def test_falls_back_to_env_var_when_secret_manager_empty(self, env):
        with patch("app.core.secret_manager.settings") as mock_settings, patch(
            "app.core.secret_manager._fetch_from_secret_manager", return_value=None
        ):
            mock_settings.ENVIRONMENT = env
            mock_settings.HUME_API_KEY = "env-fallback-key"
            assert get_hume_api_key() == "env-fallback-key"

    @pytest.mark.parametrize("env", ["staging", "production"])
    def test_raises_runtime_error_when_neither_available(self, env):
        with patch("app.core.secret_manager.settings") as mock_settings, patch(
            "app.core.secret_manager._fetch_from_secret_manager", return_value=None
        ):
            mock_settings.ENVIRONMENT = env
            mock_settings.HUME_API_KEY = ""
            with pytest.raises(RuntimeError, match="HUME_API_KEY unavailable"):
                get_hume_api_key()

    def test_environment_lookup_is_case_insensitive(self):
        with patch("app.core.secret_manager.settings") as mock_settings, patch(
            "app.core.secret_manager._fetch_from_secret_manager",
            return_value="secret-manager-hume-key",
        ):
            mock_settings.ENVIRONMENT = "STAGING"
            mock_settings.HUME_API_KEY = ""
            assert get_hume_api_key() == "secret-manager-hume-key"
