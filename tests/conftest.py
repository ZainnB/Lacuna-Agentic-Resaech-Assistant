"""Shared fixtures. Unit tests never touch the network or Docker; `live` tests skip when a service
is missing rather than fail."""

import os

import pytest

from lacuna.cache import MemoryCache


@pytest.fixture
def memory_cache():
    return MemoryCache()


@pytest.fixture
def settings(monkeypatch, tmp_path):
    """A fresh Settings object with no API keys, isolated from the developer's .env."""
    from lacuna import settings as s

    for var in list(os.environ):
        if var.endswith("_API_KEY") or var in ("SPEND_STAGE", "CONTACT_EMAIL"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(s.Settings, "model_config", {**s.Settings.model_config, "env_file": None})
    s.get_settings.cache_clear()
    yield s.get_settings()
    s.get_settings.cache_clear()


@pytest.fixture(scope="session")
def pg_conn():
    """Live Postgres from docker compose, with the schema applied."""
    from lacuna.db import connect, migrate

    try:
        conn = connect()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Postgres unavailable: {e}")
    migrate(conn)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def neo4j_driver():
    """Live Neo4j from docker compose."""
    from neo4j import GraphDatabase

    from lacuna.settings import get_settings

    cfg = get_settings()
    try:
        driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password))
        driver.verify_connectivity()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Neo4j unavailable: {e}")
    yield driver
    driver.close()
