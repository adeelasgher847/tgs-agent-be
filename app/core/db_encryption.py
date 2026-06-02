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
"""

from __future__ import annotations

import base64

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings

# First byte of common OpenPGP packets emitted by pgp_sym_encrypt (sym-enc / compressed).
_PGP_SYM_ENCRYPT_MARKERS = frozenset({0x85, 0x8C, 0xC3, 0xD3})


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
    result = db.execute(
        text("SELECT encode(pgp_sym_encrypt(:pt, :key), 'base64')"),
        {"pt": plaintext, "key": key},
    ).scalar()
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
        result = db.execute(
            text("SELECT pgp_sym_decrypt(decode(:ct, 'base64'), :key)"),
            {"ct": ciphertext, "key": key},
        ).scalar()
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
    """Return True if *value* looks like pgp_sym_encrypt base64 ciphertext."""
    if not value or is_legacy_jwt_ciphertext(value):
        return False
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception:
        return False
    return bool(raw) and raw[0] in _PGP_SYM_ENCRYPT_MARKERS


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
        raise ValueError(
            "Unrecognized ElevenLabs key ciphertext format "
            "(expected legacy JWT or pgcrypto base64)."
        )

    if db is not None:
        return decrypt_elevenlabs_key(ciphertext, db)

    from app.db.session import SessionLocal

    _db = SessionLocal()
    try:
        return decrypt_elevenlabs_key(ciphertext, _db)
    finally:
        _db.close()
