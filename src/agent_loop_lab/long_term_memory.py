"""Structured, owner-scoped long-term memory with deterministic retrieval."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Iterator, Protocol, Sequence
from uuid import uuid4


MEMORY_CATEGORIES = frozenset({"fact", "preference", "goal", "constraint", "summary"})


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    owner_id: str
    category: str
    content: str
    memory_id: str = ""
    source_event_ids: tuple[str, ...] = ()
    confidence: float = 1.0
    created_at: str = ""
    expires_at: str | None = None

    def __post_init__(self) -> None:
        _validate_owner(self.owner_id)
        if self.category not in MEMORY_CATEGORIES:
            raise ValueError(f"category must be one of: {', '.join(sorted(MEMORY_CATEGORIES))}")
        if not self.content.strip() or len(self.content) > 2_000:
            raise ValueError("content must contain 1-2000 characters")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.memory_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.memory_id):
            raise ValueError("memory_id must contain 1-64 letters, digits, _ or -")
        if len(self.source_event_ids) > 20:
            raise ValueError("source_event_ids cannot contain more than 20 items")
        if any(
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", event_id)
            for event_id in self.source_event_ids
        ):
            raise ValueError("source_event_ids contain an invalid identifier")
        if self.created_at:
            _parse_timestamp(self.created_at)
        if self.expires_at is not None:
            _parse_timestamp(self.expires_at)

    def normalized(self) -> "MemoryRecord":
        return replace(
            self,
            content=self.content.strip(),
            memory_id=self.memory_id or uuid4().hex,
            created_at=self.created_at or _utc_now(),
        )

    def render(self) -> str:
        return f"[{self.category}; confidence={self.confidence:.2f}] {self.content}"


class LongTermMemoryStore(Protocol):
    def remember(self, record: MemoryRecord) -> MemoryRecord: ...

    def search(self, owner_id: str, query: str, *, limit: int = 5) -> tuple[MemoryRecord, ...]: ...

    def list(self, owner_id: str, *, limit: int = 20) -> tuple[MemoryRecord, ...]: ...

    def forget(self, owner_id: str, memory_id: str) -> bool: ...


class InMemoryLongTermMemoryStore:
    def __init__(self, *, max_memories: int = 10_000) -> None:
        if max_memories < 1:
            raise ValueError("max_memories must be positive")
        self._max_memories = max_memories
        self._records: OrderedDict[tuple[str, str], MemoryRecord] = OrderedDict()
        self._lock = RLock()

    def remember(self, record: MemoryRecord) -> MemoryRecord:
        normalized = record.normalized()
        with self._lock:
            key = (normalized.owner_id, normalized.memory_id)
            if key in self._records:
                raise ValueError(f"memory already exists: {normalized.memory_id}")
            self._records[key] = normalized
            while len(self._records) > self._max_memories:
                self._records.popitem(last=False)
        return normalized

    def search(self, owner_id: str, query: str, *, limit: int = 5) -> tuple[MemoryRecord, ...]:
        _validate_limit(limit)
        _validate_owner(owner_id)
        with self._lock:
            records = tuple(self._records.values())
        return _rank(records, owner_id, query, limit)

    def list(self, owner_id: str, *, limit: int = 20) -> tuple[MemoryRecord, ...]:
        return self.search(owner_id, "", limit=limit)

    def forget(self, owner_id: str, memory_id: str) -> bool:
        _validate_owner(owner_id)
        with self._lock:
            key = (owner_id, memory_id)
            record = self._records.get(key)
            if record is None:
                return False
            del self._records[key]
            return True


class SQLiteLongTermMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._initialize()

    def remember(self, record: MemoryRecord) -> MemoryRecord:
        normalized = record.normalized()
        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO long_term_memories(
                        memory_id, owner_id, category, content, source_event_ids,
                        confidence, created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized.memory_id,
                        normalized.owner_id,
                        normalized.category,
                        normalized.content,
                        "\n".join(normalized.source_event_ids),
                        normalized.confidence,
                        normalized.created_at,
                        normalized.expires_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"memory already exists: {normalized.memory_id}") from exc
        return normalized

    def search(self, owner_id: str, query: str, *, limit: int = 5) -> tuple[MemoryRecord, ...]:
        _validate_limit(limit)
        _validate_owner(owner_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_id, owner_id, category, content, source_event_ids,
                       confidence, created_at, expires_at
                FROM long_term_memories
                WHERE owner_id = ?
                """,
                (owner_id,),
            ).fetchall()
        records = tuple(self._decode(row) for row in rows)
        return _rank(records, owner_id, query, limit)

    def list(self, owner_id: str, *, limit: int = 20) -> tuple[MemoryRecord, ...]:
        return self.search(owner_id, "", limit=limit)

    def forget(self, owner_id: str, memory_id: str) -> bool:
        _validate_owner(owner_id)
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM long_term_memories WHERE owner_id = ? AND memory_id = ?",
                (owner_id, memory_id),
            )
        return cursor.rowcount > 0

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    memory_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_event_ids TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    PRIMARY KEY (owner_id, memory_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_owner ON long_term_memories(owner_id)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, timeout=5.0)
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _decode(row: Sequence[object]) -> MemoryRecord:
        source_ids = tuple(str(row[4]).splitlines()) if row[4] else ()
        return MemoryRecord(
            memory_id=str(row[0]),
            owner_id=str(row[1]),
            category=str(row[2]),
            content=str(row[3]),
            source_event_ids=source_ids,
            confidence=float(row[5]),
            created_at=str(row[6]),
            expires_at=str(row[7]) if row[7] is not None else None,
        )


def _rank(
    records: Sequence[MemoryRecord],
    owner_id: str,
    query: str,
    limit: int,
) -> tuple[MemoryRecord, ...]:
    now = datetime.now(timezone.utc)
    live = [
        record
        for record in records
        if record.owner_id == owner_id
        and (record.expires_at is None or _parse_timestamp(record.expires_at) > now)
    ]
    terms = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", query.casefold()))

    def score(record: MemoryRecord) -> tuple[float, str]:
        haystack = f"{record.category} {record.content}".casefold()
        matches = sum(term in haystack for term in terms)
        return (matches * record.confidence, record.created_at)

    if terms:
        live = [record for record in live if score(record)[0] > 0]
    live.sort(key=score, reverse=True)
    return tuple(live[:limit])


def _validate_owner(owner_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", owner_id):
        raise ValueError("owner_id must contain 1-64 letters, digits, _ or -")


def _validate_limit(limit: int) -> None:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)
