"""Unit tests for app.core.db_encryption.encrypt_webhook_headers /
decrypt_webhook_headers (System Webhooks per-Call-Flow header columns).

Mocks the pgcrypto SQL round-trip (db.execute(...).scalar()) rather than
hitting a real Postgres — mirrors tests/api/v2/test_webhooks.py's
TestSecretEncryption pattern for the sibling encrypt_webhook_secret/
decrypt_webhook_secret pair.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


class TestEncryptWebhookHeaders:
    def test_round_trip_with_real_dict(self):
        """Simulates the full round trip by feeding decrypt the exact
        plaintext encrypt would have produced (via a MagicMock DB standing in
        for Postgres's pgp_sym_encrypt/pgp_sym_decrypt)."""
        from app.core.db_encryption import (
            decrypt_webhook_headers,
            encrypt_webhook_headers,
        )

        headers = {"Authorization": "Bearer secret-token", "X-Custom": "value"}

        db = MagicMock()
        db.execute.return_value.scalar.return_value = "fake-ciphertext-b64"

        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.WEBHOOK_SECRET_ENCRYPTION_KEY = (
                "test-key-32-chars-long-pad12345"
            )
            ciphertext = encrypt_webhook_headers(headers, db)

            assert ciphertext == "fake-ciphertext-b64"
            # Confirm the plaintext handed to pgcrypto was the JSON-serialized dict.
            call_kwargs = db.execute.call_args
            bound_params = call_kwargs[0][1]
            assert json.loads(bound_params["pt"]) == headers

            # Now decrypt: stub pgcrypto's decrypt leg to return that same JSON.
            db.execute.return_value.scalar.return_value = json.dumps(headers)
            result = decrypt_webhook_headers(ciphertext, db)

        assert result == headers

    def test_empty_dict_returns_none_and_skips_encryption(self):
        """Per the documented contract: a falsy/empty headers dict must not be
        encrypted — callers should store NULL, not ciphertext of '{}'."""
        from app.core.db_encryption import encrypt_webhook_headers

        db = MagicMock()
        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.WEBHOOK_SECRET_ENCRYPTION_KEY = (
                "test-key-32-chars-long-pad12345"
            )
            result = encrypt_webhook_headers({}, db)

        assert result is None
        db.execute.assert_not_called()

    def test_none_headers_returns_none_and_skips_encryption(self):
        from app.core.db_encryption import encrypt_webhook_headers

        db = MagicMock()
        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.WEBHOOK_SECRET_ENCRYPTION_KEY = (
                "test-key-32-chars-long-pad12345"
            )
            result = encrypt_webhook_headers(None, db)

        assert result is None
        db.execute.assert_not_called()

    def test_encrypt_raises_without_key(self):
        from app.core.db_encryption import encrypt_webhook_headers

        db = MagicMock()
        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.WEBHOOK_SECRET_ENCRYPTION_KEY = ""
            with pytest.raises(ValueError, match="WEBHOOK_SECRET_ENCRYPTION_KEY"):
                encrypt_webhook_headers({"a": "b"}, db)

    def test_decrypt_raises_without_key(self):
        from app.core.db_encryption import decrypt_webhook_headers

        db = MagicMock()
        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.WEBHOOK_SECRET_ENCRYPTION_KEY = ""
            with pytest.raises(ValueError, match="WEBHOOK_SECRET_ENCRYPTION_KEY"):
                decrypt_webhook_headers("some-ciphertext", db)

    def test_decrypt_empty_pgcrypto_result_returns_empty_dict(self):
        from app.core.db_encryption import decrypt_webhook_headers

        db = MagicMock()
        db.execute.return_value.scalar.return_value = None
        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.WEBHOOK_SECRET_ENCRYPTION_KEY = (
                "test-key-32-chars-long-pad12345"
            )
            result = decrypt_webhook_headers("some-ciphertext", db)

        assert result == {}

    def test_decrypt_invalid_json_raises_value_error(self):
        from app.core.db_encryption import decrypt_webhook_headers

        db = MagicMock()
        db.execute.return_value.scalar.return_value = "not-json{{{"
        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.WEBHOOK_SECRET_ENCRYPTION_KEY = (
                "test-key-32-chars-long-pad12345"
            )
            with pytest.raises(ValueError, match="invalid JSON"):
                decrypt_webhook_headers("some-ciphertext", db)

    def test_decrypt_non_dict_json_raises_value_error(self):
        from app.core.db_encryption import decrypt_webhook_headers

        db = MagicMock()
        db.execute.return_value.scalar.return_value = json.dumps(["not", "a", "dict"])
        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.WEBHOOK_SECRET_ENCRYPTION_KEY = (
                "test-key-32-chars-long-pad12345"
            )
            with pytest.raises(ValueError, match="JSON object"):
                decrypt_webhook_headers("some-ciphertext", db)
