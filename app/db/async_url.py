"""Build SQLAlchemy asyncpg URLs from sync psycopg2-style DATABASE_URL values."""
from __future__ import annotations

from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# libpq / psycopg2 query params that asyncpg does not accept
_STRIP_QUERY_KEYS = frozenset({"sslmode", "channel_binding", "options", "gssencmode"})

_SSLMODE_TO_ASYNCPG = {
    "disable": None,
    "allow": "prefer",
    "prefer": "prefer",
    "require": "require",
    "verify-ca": "require",
    "verify-full": "require",
}


def database_url_to_async(database_url: str) -> str:
    """
    Convert a sync Postgres URL to ``postgresql+asyncpg://`` with compatible query params.

    Maps ``sslmode`` (psycopg2/libpq) to ``ssl`` (asyncpg). Strips other libpq-only keys.
    """
    url = database_url.strip()
    if url.startswith("postgresql+psycopg2://"):
        url = "postgresql+asyncpg://" + url[len("postgresql+psycopg2://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://") :]
    elif not url.startswith("postgresql+asyncpg://"):
        raise ValueError(f"Unsupported database URL scheme: {database_url!r}")

    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)

    if "sslmode" in query:
        sslmode = (query.pop("sslmode")[0] or "prefer").lower()
        ssl_value = _SSLMODE_TO_ASYNCPG.get(sslmode, "require")
        if ssl_value is not None:
            query["ssl"] = [ssl_value]

    for key in _STRIP_QUERY_KEYS:
        query.pop(key, None)

    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=new_query))
