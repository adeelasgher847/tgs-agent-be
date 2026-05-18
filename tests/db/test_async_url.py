import pytest

from app.db.async_url import database_url_to_async


def test_psycopg2_sslmode_maps_to_asyncpg_ssl():
    url = (
        "postgresql+psycopg2://user:pass@host/db"
        "?sslmode=require&channel_binding=require"
    )
    out = database_url_to_async(url)
    assert out.startswith("postgresql+asyncpg://")
    assert "sslmode" not in out
    assert "channel_binding" not in out
    assert "ssl=require" in out


def test_plain_postgresql_url():
    url = "postgresql://user:pass@localhost:5432/voiceagent"
    out = database_url_to_async(url)
    assert out == "postgresql+asyncpg://user:pass@localhost:5432/voiceagent"


def test_sslmode_disable_omits_ssl_param():
    url = "postgresql+psycopg2://user:pass@localhost/db?sslmode=disable"
    out = database_url_to_async(url)
    assert "ssl=" not in out
    assert "sslmode" not in out
