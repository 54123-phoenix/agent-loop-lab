"""Bounded in-memory conversation storage for the HTTP layer."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Iterator, Protocol, Sequence

from .models import Message, ToolCall


class ConversationStore(Protocol):
    def load(self, session_id: str) -> tuple[Message, ...]: ...

    def save(self, session_id: str, messages: Sequence[Message]) -> None: ...

    def clear(self, session_id: str) -> bool: ...


class InMemoryConversationStore:
    """LRU session store with a fixed message window; data is process-local."""

    def __init__(self, *, max_sessions: int = 1000, max_messages: int = 40) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        self._max_sessions = max_sessions
        self._max_messages = max_messages
        self._sessions: OrderedDict[str, tuple[Message, ...]] = OrderedDict()
        self._lock = RLock()

    def load(self, session_id: str) -> tuple[Message, ...]:
        self._validate_session_id(session_id)
        with self._lock:
            messages = self._sessions.get(session_id, ())
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
            return messages

    def save(self, session_id: str, messages: Sequence[Message]) -> None:
        self._validate_session_id(session_id)
        window = self._safe_window(messages)
        with self._lock:
            self._sessions[session_id] = window
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

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
        InMemoryConversationStore._validate_session_id(session_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM conversation_events
                WHERE session_id = ?
                ORDER BY sequence
                """,
                (session_id,),
            ).fetchall()
        return tuple(self._decode_message(row[0]) for row in rows)

    def save(self, session_id: str, messages: Sequence[Message]) -> None:
        InMemoryConversationStore._validate_session_id(session_id)
        incoming = tuple(messages)
        existing = self.load(session_id)
        if incoming[: len(existing)] != existing:
            raise ValueError("messages must extend the persisted session history")
        new_messages = incoming[len(existing) :]
        if not new_messages:
            return
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
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

    def clear(self, session_id: str) -> bool:
        InMemoryConversationStore._validate_session_id(session_id)
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversation_events WHERE session_id = ?",
                (session_id,),
            )
        return cursor.rowcount > 0

    def _initialize(self) -> None:
        with self._connect() as connection:
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
        return Message(
            role=payload["role"],
            content=payload["content"],
            name=payload.get("name"),
            tool_call=tool_call,
            call_id=payload.get("call_id"),
        )
