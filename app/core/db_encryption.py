"""pgcrypto-backed symmetric encryption for sensitive DB columns.

Uses PostgreSQL's ``pgp_sym_encrypt`` / ``pgp_sym_decrypt`` functions so that
the plaintext never crosses the Python ↔ Postgres wire in a recoverable form.
The ciphertext is stored as base64-encoded TEXT to keep the column type
unchanged.

Key configuration
-----------------
Set ``ELEVENLABS_ENCRYPTION_KEY`` in the environment (or ``.env``).  In
staging / production this should come from Secret Manager via the same pattern
used by the Twilio credentials.

If the key is missing or empty the encrypt call raises ``ValueError`` so that
the application fails loudly rather than silently storing plaintext.

HubSpot tokens (below) are the one exception: pgcrypto's ``pgp_sym_encrypt``
is OpenPGP symmetric encryption and has no GCM mode, so it cannot satisfy a
literal AES-256-GCM requirement. Those two helpers perform AES-256-GCM in
Python via ``cryptography`` instead of pgcrypto SQL.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from contextlib import contextmanager

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings

_log = logging.getLogger(__name__)

# Best-effort first-byte heuristic for pgp_sym_encrypt output (sym-enc / compressed).
# False positives are possible: arbitrary valid base64 whose first decoded byte matches
# an OpenPGP packet tag will be routed to pgcrypto decrypt and surface as
# "ElevenLabs key decryption failed" instead of "unrecognized format".
_PGP_SYM_ENCRYPT_MARKERS = frozenset({0x85, 0x8C, 0xC3, 0xD3})


@contextmanager
def _suppress_pgcrypto_sql_logging(db: Session):
    """Prevent :key bound params from appearing in SQLAlchemy engine logs."""
    # WARNING: never enable SQLALCHEMY_ECHO in staging/production — the :key parameter
    # would appear in logs. Use ENVIRONMENT-specific log level controls instead.
    engine = db.get_bind()
    sql_logger = logging.getLogger("sqlalchemy.engine")
    prev_echo, prev_level = engine.echo, sql_logger.level
    try:
        engine.echo = False
        sql_logger.setLevel(logging.WARNING)
        yield
    finally:
        engine.echo = prev_echo
        sql_logger.setLevel(prev_level)


def _pgcrypto_scalar(db: Session, sql: str, params: dict[str, str]) -> str | None:
    """Run a pgcrypto statement without logging :key bound parameters."""
    with _suppress_pgcrypto_sql_logging(db):
        return db.execute(text(sql), params).scalar()


def encrypt_elevenlabs_key(plaintext: str, db: Session) -> str:
    """Encrypt *plaintext* using pgp_sym_encrypt; return base64 ciphertext.

    Raises ``ValueError`` if ``ELEVENLABS_ENCRYPTION_KEY`` is not configured.
    """
    key = settings.ELEVENLABS_ENCRYPTION_KEY
    if not key:
        raise ValueError(
            "ELEVENLABS_ENCRYPTION_KEY is not configured — "
            "cannot encrypt ElevenLabs API key."
        )
    result = _pgcrypto_scalar(
        db,
        "SELECT encode(pgp_sym_encrypt(:pt, :key), 'base64')",
        {"pt": plaintext, "key": key},
    )
    return result  # type: ignore[return-value]


def decrypt_elevenlabs_key(ciphertext: str, db: Session) -> str:
    """Decrypt base64 ciphertext produced by :func:`encrypt_elevenlabs_key`.

    Raises ``ValueError`` if ``ELEVENLABS_ENCRYPTION_KEY`` is not configured or
    if the ciphertext is corrupt / encrypted with a different key.
    """
    key = settings.ELEVENLABS_ENCRYPTION_KEY
    if not key:
        raise ValueError(
            "ELEVENLABS_ENCRYPTION_KEY is not configured — "
            "cannot decrypt ElevenLabs API key."
        )
    try:
        result = _pgcrypto_scalar(
            db,
            "SELECT pgp_sym_decrypt(decode(:ct, 'base64'), :key)",
            {"ct": ciphertext, "key": key},
        )
        return result or ""  # type: ignore[return-value]
    except Exception as exc:
        raise ValueError(f"ElevenLabs key decryption failed: {exc}") from exc


def is_legacy_jwt_ciphertext(value: str) -> bool:
    """Return True if *value* is a compact JWS (legacy encrypt_api_key output)."""
    if not value or not value.startswith("eyJ"):
        return False
    parts = value.split(".")
    return len(parts) == 3 and all(parts)


def is_pgcrypto_ciphertext(value: str) -> bool:
    """Return True if *value* looks like pgp_sym_encrypt base64 ciphertext.

    Best-effort only; see :data:`_PGP_SYM_ENCRYPT_MARKERS` for false-positive risk.
    """
    if not value or is_legacy_jwt_ciphertext(value):
        return False
    try:
        # PostgreSQL encode(..., 'base64') includes newline characters, so strip them
        cleaned_value = "".join(value.split())
        raw = base64.b64decode(cleaned_value, validate=True)
    except Exception:
        return False
    return bool(raw) and raw[0] in _PGP_SYM_ENCRYPT_MARKERS


def encrypt_webhook_secret(plaintext: str, db: Session) -> str:
    """Encrypt *plaintext* webhook secret using pgp_sym_encrypt; return base64 ciphertext.

    Raises ``ValueError`` if ``WEBHOOK_SECRET_ENCRYPTION_KEY`` is not configured.
    """
    key = settings.WEBHOOK_SECRET_ENCRYPTION_KEY
    if not key:
        raise ValueError(
            "WEBHOOK_SECRET_ENCRYPTION_KEY is not configured — "
            "cannot encrypt webhook secret."
        )
    result = _pgcrypto_scalar(
        db,
        "SELECT encode(pgp_sym_encrypt(:pt, :key), 'base64')",
        {"pt": plaintext, "key": key},
    )
    return result  # type: ignore[return-value]


def decrypt_webhook_secret(ciphertext: str, db: Session) -> str:
    """Decrypt base64 ciphertext produced by :func:`encrypt_webhook_secret`.

    Raises ``ValueError`` if ``WEBHOOK_SECRET_ENCRYPTION_KEY`` is not configured or
    if the ciphertext is corrupt / encrypted with a different key.
    """
    key = settings.WEBHOOK_SECRET_ENCRYPTION_KEY
    if not key:
        raise ValueError(
            "WEBHOOK_SECRET_ENCRYPTION_KEY is not configured — "
            "cannot decrypt webhook secret."
        )
    try:
        result = _pgcrypto_scalar(
            db,
            "SELECT pgp_sym_decrypt(decode(:ct, 'base64'), :key)",
            {"ct": ciphertext, "key": key},
        )
        return result or ""  # type: ignore[return-value]
    except Exception as exc:
        raise ValueError(f"Webhook secret decryption failed: {exc}") from exc


def decrypt_stored_webhook_secret(
    ciphertext: str,
    *,
    db: "Session | None" = None,
) -> str:
    """Unified webhook secret decrypt — handles both pgcrypto and legacy JWT.

    New secrets are always written as pgcrypto (base64 PGP) via
    :func:`encrypt_webhook_secret`.  Secrets written before v20260612 were
    JWT-encrypted via ``encrypt_api_key``; those are decrypted transparently
    so existing rows keep working without a data migration.

    Raises ``ValueError`` on unrecognisable ciphertext or missing config.
    """
    if not ciphertext:
        raise ValueError("ciphertext is empty")

    if is_legacy_jwt_ciphertext(ciphertext):
        from app.core.security import decrypt_api_key

        return decrypt_api_key(ciphertext)

    if not is_pgcrypto_ciphertext(ciphertext):
        _log.warning(
            "Webhook secret: unrecognized ciphertext format "
            "(expected legacy JWT or pgcrypto base64)"
        )
        raise ValueError(
            "Unrecognized webhook secret ciphertext format "
            "(expected legacy JWT or pgcrypto base64)."
        )

    try:
        if db is not None:
            return decrypt_webhook_secret(ciphertext, db)

        from app.db.session import SessionLocal

        _db = SessionLocal()
        try:
            return decrypt_webhook_secret(ciphertext, _db)
        finally:
            _db.close()
    except ValueError as exc:
        _log.warning(
            "Webhook secret: pgcrypto decrypt failed (%s) — "
            "wrong WEBHOOK_SECRET_ENCRYPTION_KEY or heuristic false-positive",
            exc,
        )
        raise


# Tags new-format ciphertext unambiguously so decrypt never has to guess between
# AES-256-GCM and the legacy pgp_sym_encrypt format below — no byte-sniffing heuristic.
_HUBSPOT_AESGCM_PREFIX = "gcm1:"


def _hubspot_aes_key() -> bytes:
    """Derive a 32-byte AES-256 key from HUBSPOT_TOKEN_ENCRYPTION_KEY.

    SECURITY NOTE:
      For secure AES-256-GCM encryption, HUBSPOT_TOKEN_ENCRYPTION_KEY must be configured
      as a randomly generated 32-byte secret (typically represented as a 64-character
      hex string, e.g. generated via `secrets.token_hex(32)`).

    BACKWARD COMPATIBILITY:
      We derive the key using hashlib.sha256 to ensure that any key length/format can
      be safely resolved, and to preserve 100% backward compatibility for already
      connected HubSpot workspaces encrypted under older/existing settings keys.
    """
    key = settings.HUBSPOT_TOKEN_ENCRYPTION_KEY
    if not key:
        raise ValueError(
            "HUBSPOT_TOKEN_ENCRYPTION_KEY is not configured — "
            "cannot encrypt/decrypt HubSpot OAuth token."
        )
    return hashlib.sha256(key.encode("utf-8")).digest()


def encrypt_hubspot_token(plaintext: str, db: Session) -> str:  # noqa: ARG001 - db kept for call-site parity
    """Encrypt *plaintext* HubSpot OAuth token with AES-256-GCM.

    Performed in Python via ``cryptography`` (not pgcrypto SQL — OpenPGP symmetric
    encryption has no GCM mode). ``db`` is accepted only for parity with the other
    encrypt_* helpers in this module; it is unused here.

    Raises ``ValueError`` if ``HUBSPOT_TOKEN_ENCRYPTION_KEY`` is not configured.
    """
    key = _hubspot_aes_key()
    nonce = os.urandom(12)  # 96-bit nonce, the standard size for AES-GCM
    ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    return _HUBSPOT_AESGCM_PREFIX + base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt_hubspot_token(ciphertext: str, db: Session) -> str:
    """Decrypt a HubSpot OAuth token written by :func:`encrypt_hubspot_token`.

    Handles two formats so workspaces connected before the AES-256-GCM migration
    don't need to reconnect:
      - ``gcm1:``-prefixed: AES-256-GCM (current format).
      - Unprefixed: legacy pgp_sym_encrypt base64 (pgcrypto), written before this
        migration — decrypted via the same SQL path as the ElevenLabs/webhook keys.

    Raises ``ValueError`` if ``HUBSPOT_TOKEN_ENCRYPTION_KEY`` is not configured or
    if the ciphertext is corrupt / encrypted with a different key.
    """
    if not ciphertext:
        raise ValueError("ciphertext is empty")

    if ciphertext.startswith(_HUBSPOT_AESGCM_PREFIX):
        key = _hubspot_aes_key()
        try:
            raw = base64.b64decode(ciphertext[len(_HUBSPOT_AESGCM_PREFIX):])
            nonce, body = raw[:12], raw[12:]
            return AESGCM(key).decrypt(nonce, body, None).decode("utf-8")
        except Exception as exc:
            raise ValueError(f"HubSpot token decryption failed: {exc}") from exc

    # Legacy pgp_sym_encrypt ciphertext (pre-AES-256-GCM migration).
    key = settings.HUBSPOT_TOKEN_ENCRYPTION_KEY
    if not key:
        raise ValueError(
            "HUBSPOT_TOKEN_ENCRYPTION_KEY is not configured — "
            "cannot decrypt HubSpot OAuth token."
        )
    try:
        result = _pgcrypto_scalar(
            db,
            "SELECT pgp_sym_decrypt(decode(:ct, 'base64'), :key)",
            {"ct": ciphertext, "key": key},
        )
        return result or ""  # type: ignore[return-value]
    except Exception as exc:
        raise ValueError(f"HubSpot token decryption failed: {exc}") from exc


def decrypt_stored_elevenlabs_key(
    ciphertext: str,
    *,
    db: "Session | None" = None,
) -> str:
    """Unified BYO ElevenLabs key decrypt — handles both pgcrypto and legacy JWT.

    Call this everywhere a stored ``agent.encrypted_elevenlabs_api_key`` needs
    to be turned back into plaintext.  It automatically detects which format the
    ciphertext is in so that rows encrypted before the v2 migration (JWT) and
    rows encrypted after (pgcrypto) both work until all rows are re-encrypted.

    Detection logic
    ---------------
    - :func:`is_legacy_jwt_ciphertext` → :func:`app.core.security.decrypt_api_key`
    - :func:`is_pgcrypto_ciphertext`   → :func:`decrypt_elevenlabs_key`
    - Neither                          → ``ValueError`` (no blind pgp_sym_decrypt attempt)

    ``db`` parameter
    ----------------
    Required for pgcrypto decryption.  If *db* is ``None`` and the ciphertext is
    a pgcrypto blob a short-lived :class:`~app.db.session.SessionLocal` is opened
    for the single SQL call and closed immediately.  Prefer passing the caller's
    existing session to avoid extra connection overhead.

    Raises ``ValueError`` on unrecognisable ciphertext or missing config.
    """
    if not ciphertext:
        raise ValueError("ciphertext is empty")

    if is_legacy_jwt_ciphertext(ciphertext):
        from app.core.security import decrypt_api_key
        return decrypt_api_key(ciphertext)

    if not is_pgcrypto_ciphertext(ciphertext):
        _log.warning(
            "ElevenLabs stored key: unrecognized ciphertext format "
            "(expected legacy JWT or pgcrypto base64)"
        )
        raise ValueError(
            "Unrecognized ElevenLabs key ciphertext format "
            "(expected legacy JWT or pgcrypto base64)."
        )

    try:
        if db is not None:
            return decrypt_elevenlabs_key(ciphertext, db)

        from app.db.session import SessionLocal

        _db = SessionLocal()
        try:
            return decrypt_elevenlabs_key(ciphertext, _db)
        finally:
            _db.close()
    except ValueError as exc:
        _log.warning(
            "ElevenLabs stored key: pgcrypto decrypt failed (%s) — "
            "wrong ELEVENLABS_ENCRYPTION_KEY or heuristic false-positive",
            exc,
        )
        raise
