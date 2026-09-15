"""Bounded in-memory conversation storage for the HTTP layer."""

from __future__ import annotations

from collections import OrderedDict
import re
from threading import RLock
from typing import Protocol, Sequence

from .models import Message


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
