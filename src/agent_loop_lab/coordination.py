"""Per-session coordination and request-idempotency helpers."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import AsyncIterator, Iterator, Mapping, Protocol


class RequestJournal(Protocol):
    def get(self, session_id: str, request_id: str) -> Mapping[str, object] | None: ...

    def put(
        self,
        session_id: str,
        request_id: str,
        response: Mapping[str, object],
    ) -> None: ...


class InMemoryRequestJournal:
    def __init__(self) -> None:
        self._responses: dict[tuple[str, str], dict[str, object]] = {}
        self._lock = RLock()

    def get(self, session_id: str, request_id: str) -> Mapping[str, object] | None:
        with self._lock:
            response = self._responses.get((session_id, request_id))
            return dict(response) if response is not None else None

    def put(
        self,
        session_id: str,
        request_id: str,
        response: Mapping[str, object],
    ) -> None:
        with self._lock:
            self._responses.setdefault((session_id, request_id), dict(response))


class SQLiteRequestJournal:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS request_results (
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, request_id)
                )
                """
            )

    def get(self, session_id: str, request_id: str) -> Mapping[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT response_json FROM request_results
                WHERE session_id = ? AND request_id = ?
                """,
                (session_id, request_id),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(
        self,
        session_id: str,
        request_id: str,
        response: Mapping[str, object],
    ) -> None:
        payload = json.dumps(dict(response), ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO request_results(session_id, request_id, response_json)
                VALUES (?, ?, ?)
                """,
                (session_id, request_id, payload),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, timeout=5.0)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


class SessionCoordinator:
    """Serialize work for one session while allowing different sessions in parallel."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def hold(self, session_id: str) -> AsyncIterator[None]:
        async with self._guard:
            lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            yield
