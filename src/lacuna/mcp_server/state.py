"""Per-process state: one Postgres connection, one GraphRepo, one session id. Both are lazy."""

from __future__ import annotations

import logging
import threading
import uuid

from lacuna import db

log = logging.getLogger("lacuna.mcp")


class State:
    def __init__(self, session: str | None = None) -> None:
        self.session = session or f"mcp-{uuid.uuid4().hex[:12]}"
        self.lock = threading.RLock()
        self._conn = None
        self._repo = None
        self._ready = False

    # -- connections -------------------------------------------------------------------------
    def conn(self):
        """The live connection; reconnects when it was closed or the server went away."""
        with self.lock:
            if self._conn is not None:
                try:
                    if self._conn.closed:
                        raise ConnectionError("closed")
                    self._conn.execute("SELECT 1")
                except Exception:
                    log.warning("postgres connection lost; reconnecting")
                    self._drop_conn()
            if self._conn is None:
                self._conn = db.connect()
                db.migrate(self._conn)
            return self._conn

    def repo(self):
        """The Neo4j repo; reconnects when connectivity check fails."""
        with self.lock:
            if self._repo is not None:
                try:
                    self._repo.driver.verify_connectivity()
                except Exception:
                    log.warning("neo4j connection lost; reconnecting")
                    self._drop_repo()
            if self._repo is None:
                from lacuna.graph.entities import ensure_entity_constraints
                from lacuna.graph.repo import GraphRepo

                repo = GraphRepo.from_settings()
                repo.ensure_constraints()
                ensure_entity_constraints(repo)
                self._repo = repo
            return self._repo

    def startup(self) -> None:
        """Best effort at process start: schema and constraints. Failures are logged, not fatal."""
        for name, fn in (("postgres", self.conn), ("neo4j", self.repo)):
            try:
                fn()
            except Exception as e:  # the server must still list its tools when a service is down
                log.warning("startup: %s unavailable: %s", name, e)

    def _drop_conn(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    def _drop_repo(self) -> None:
        try:
            if self._repo is not None:
                self._repo.close()
        except Exception:
            pass
        self._repo = None

    def close(self) -> None:
        with self.lock:
            self._drop_conn()
            self._drop_repo()

    # -- served chunks -----------------------------------------------------------------------
    def mark_served(self, conn, chunk_ids: list[str]) -> None:
        for cid in chunk_ids:
            conn.execute(
                "INSERT INTO served_chunks (chunk_id, session) VALUES (%s, %s) "
                "ON CONFLICT (chunk_id, session) DO UPDATE SET served_at = now()",
                (cid, self.session),
            )
