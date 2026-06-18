"""Origin / domain normalization shared by allowed-domains CRUD and the
public SDK token endpoint — both must agree on the same canonical form so
a stored domain matches a browser's ``Origin`` header.

Rules: lowercase, drop any path (Origin headers never have one anyway),
drop the port when it's the default HTTPS port 443.
"""
from __future__ import annotations

from urllib.parse import urlsplit


def normalize_origin(value: str) -> str:
    parsed = urlsplit(value.strip())
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port
    netloc = host if not port or port == 443 else f"{host}:{port}"
    return f"{scheme}://{netloc}"


def is_localhost_origin(value: str) -> bool:
    parsed = urlsplit(value.strip())
    return (parsed.hostname or "").lower() == "localhost"
