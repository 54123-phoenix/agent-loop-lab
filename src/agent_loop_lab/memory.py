"""Bounded in-memory conversation storage for the HTTP layer."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Iterator, Protocol, Sequence

from .models import Message, ToolCall, ToolError, ToolResult


class ConversationStore(Protocol):
    def load(self, session_id: str) -> tuple[Message, ...]: ...

    def save(self, session_id: str, messages: Sequence[Message]) -> None: ...

    def clear(self, session_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class ConversationSnapshot:
    messages: tuple[Message, ...]
    version: int


class ConversationConflictError(RuntimeError):
    """Raised when a caller tries to save from a stale snapshot."""


class InMemoryConversationStore:
    """LRU session store with a fixed message window; data is process-local."""

    def __init__(self, *, max_sessions: int = 1000, max_messages: int = 40) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        self._max_sessions = max_sessions
        self._max_messages = max_messages
        self._sessions: OrderedDict[str, ConversationSnapshot] = OrderedDict()
        self._lock = RLock()

    def load(self, session_id: str) -> tuple[Message, ...]:
        self._validate_session_id(session_id)
        with self._lock:
            snapshot = self._sessions.get(session_id, ConversationSnapshot((), 0))
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
            return snapshot.messages

    def save(self, session_id: str, messages: Sequence[Message]) -> None:
        self._validate_session_id(session_id)
        with self._lock:
            current = self._sessions.get(session_id, ConversationSnapshot((), 0))
            self._save_if_version_locked(session_id, messages, current.version)

    def load_snapshot(self, session_id: str) -> ConversationSnapshot:
        self._validate_session_id(session_id)
        with self._lock:
            snapshot = self._sessions.get(session_id, ConversationSnapshot((), 0))
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
            return snapshot

    def save_if_version(
        self,
        session_id: str,
        messages: Sequence[Message],
        *,
        expected_version: int,
    ) -> int:
        self._validate_session_id(session_id)
        with self._lock:
            return self._save_if_version_locked(session_id, messages, expected_version)

    def _save_if_version_locked(
        self,
        session_id: str,
        messages: Sequence[Message],
        expected_version: int,
    ) -> int:
        current = self._sessions.get(session_id, ConversationSnapshot((), 0))
        if current.version != expected_version:
            raise ConversationConflictError(
                f"session version changed from {expected_version} to {current.version}"
            )
        next_version = current.version + 1
        self._sessions[session_id] = ConversationSnapshot(
            self._safe_window(messages),
            next_version,
        )
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
        return next_version

    def clear(self, session_id: str) -> bool:
        self._validate_session_id(session_id)
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", session_id):
            raise ValueError("session_id must contain 1-64 letters, digits, _ or -")

    def _safe_window(self, messages: Sequence[Message]) -> tuple[Message, ...]:
        window = list(messages[-self._max_messages :])
        call_ids = {
            message.tool_call.call_id
            for message in window
            if message.tool_call is not None and message.tool_call.call_id
        }
        return tuple(
            message
            for message in window
            if not (
                message.role == "tool"
                and message.call_id
                and message.call_id not in call_ids
            )
        )


class SQLiteConversationStore:
    """Append-only SQLite message log that survives process restarts."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._initialize()

    def load(self, session_id: str) -> tuple[Message, ...]:
        return self.load_snapshot(session_id).messages

    def load_snapshot(self, session_id: str) -> ConversationSnapshot:
        InMemoryConversationStore._validate_session_id(session_id)
        with self._connect() as connection:
            version_row = connection.execute(
                "SELECT version FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            rows = connection.execute(
                """
                SELECT payload_json
                FROM conversation_events
                WHERE session_id = ?
                ORDER BY sequence
                """,
                (session_id,),
            ).fetchall()
        return ConversationSnapshot(
            tuple(self._decode_message(row[0]) for row in rows),
            int(version_row[0]) if version_row else 0,
        )

    def save(self, session_id: str, messages: Sequence[Message]) -> None:
        InMemoryConversationStore._validate_session_id(session_id)
        snapshot = self.load_snapshot(session_id)
        self.save_if_version(
            session_id,
            messages,
            expected_version=snapshot.version,
        )

    def save_if_version(
        self,
        session_id: str,
        messages: Sequence[Message],
        *,
        expected_version: int,
    ) -> int:
        InMemoryConversationStore._validate_session_id(session_id)
        incoming = tuple(messages)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            version_row = connection.execute(
                "SELECT version FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            current_version = int(version_row[0]) if version_row else 0
            if current_version != expected_version:
                raise ConversationConflictError(
                    f"session version changed from {expected_version} to {current_version}"
                )
            rows = connection.execute(
                """
                SELECT payload_json FROM conversation_events
                WHERE session_id = ? ORDER BY sequence
                """,
                (session_id,),
            ).fetchall()
            existing = tuple(self._decode_message(row[0]) for row in rows)
            if incoming[: len(existing)] != existing:
                raise ValueError("messages must extend the persisted session history")
            new_messages = incoming[len(existing) :]
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM conversation_events WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            sequence = int(row[0])
            for message in new_messages:
                sequence += 1
                connection.execute(
                    """
                    INSERT INTO conversation_events(session_id, sequence, event_type, payload_json)
                    VALUES (?, ?, 'message', ?)
                    """,
                    (session_id, sequence, self._encode_message(message)),
                )
            next_version = current_version + 1
            connection.execute(
                """
                INSERT INTO sessions(session_id, version) VALUES (?, ?)
                ON CONFLICT(session_id) DO UPDATE SET version = excluded.version
                """,
                (session_id, next_version),
            )
        return next_version

    def clear(self, session_id: str) -> bool:
        InMemoryConversationStore._validate_session_id(session_id)
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversation_events WHERE session_id = ?",
                (session_id,),
            )
            connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        return cursor.rowcount > 0

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_events (
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, sequence)
                )
                """
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
    def _encode_message(message: Message) -> str:
        return json.dumps(asdict(message), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _decode_message(payload_json: str) -> Message:
        payload = json.loads(payload_json)
        tool_call_payload = payload.get("tool_call")
        tool_call = ToolCall(**tool_call_payload) if tool_call_payload else None
        tool_result_payload = payload.get("tool_result")
        tool_result = None
        if tool_result_payload:
            error_payload = tool_result_payload.get("error")
            error = ToolError(**error_payload) if error_payload else None
            tool_result = ToolResult(
                name=tool_result_payload["name"],
                ok=tool_result_payload["ok"],
                content=tool_result_payload["content"],
                attempts=tool_result_payload.get("attempts", 1),
                duration_ms=tool_result_payload.get("duration_ms", 0.0),
                data=tool_result_payload.get("data"),
                error=error,
                truncated=tool_result_payload.get("truncated", False),
            )
        return Message(
            role=payload["role"],
            content=payload["content"],
            name=payload.get("name"),
            tool_call=tool_call,
            call_id=payload.get("call_id"),
            tool_result=tool_result,
        )
