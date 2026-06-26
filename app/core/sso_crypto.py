"""Fernet-based encryption for SSO secrets (OIDC client secrets)."""

from cryptography.fernet import Fernet
from app.core.secret_manager import get_sso_encryption_key

# We fetch the key dynamically at module load time so it requires
# SSO_ENCRYPTION_KEY to be present if this module is imported.
_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        key = get_sso_encryption_key()
        _fernet = Fernet(key)
    return _fernet


def encrypt_secret(plain_text: str) -> str:
    """Encrypt a plain text secret using Fernet (AES-128-CBC + HMAC-SHA256)."""
    if not plain_text:
        return plain_text
    f = _get_fernet()
    return f.encrypt(plain_text.encode("utf-8")).decode("utf-8")


def decrypt_secret(cipher_text: str) -> str:
    """Decrypt a Fernet-encrypted secret."""
    if not cipher_text:
        return cipher_text
    f = _get_fernet()
    return f.decrypt(cipher_text.encode("utf-8")).decode("utf-8")
