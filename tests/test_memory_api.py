from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from agent_loop_lab import Agent, Message, ModelResponse, build_default_registry
from agent_loop_lab.api import create_app
from agent_loop_lab.memory import InMemoryConversationStore


class HistoryAwareModel:
    def respond(self, messages, tools):
        del tools
        return ModelResponse.final(f"seen={len(messages)}")


class MemoryTests(unittest.TestCase):
    def test_store_keeps_bounded_message_window(self) -> None:
        store = InMemoryConversationStore(max_messages=2)
        store.save(
            "session",
            [
                Message("user", "one"),
                Message("assistant", "two"),
                Message("user", "three"),
            ],
        )

        self.assertEqual(
            [message.content for message in store.load("session")],
            ["two", "three"],
        )

    def test_store_evicts_oldest_session(self) -> None:
        store = InMemoryConversationStore(max_sessions=1)
        store.save("one", [Message("user", "one")])
        store.save("two", [Message("user", "two")])

        self.assertEqual(store.load("one"), ())
        self.assertEqual(len(store.load("two")), 1)

    def test_store_removes_orphaned_function_output(self) -> None:
        store = InMemoryConversationStore(max_messages=2)
        store.save(
            "session",
            [
                Message("assistant", "call", call_id="call_1"),
                Message("tool", "result", "calculator", call_id="call_1"),
                Message("assistant", "done"),
            ],
        )

        self.assertEqual(
            [message.content for message in store.load("session")],
            ["done"],
        )


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        store = InMemoryConversationStore(max_messages=10)
        app = create_app(
            lambda: Agent(HistoryAwareModel(), build_default_registry()),
            store=store,
        )
        self.client = TestClient(app)

    def test_chat_reuses_session_history(self) -> None:
        first = self.client.post(
            "/v1/chat",
            json={"session_id": "demo", "message": "first"},
        )
        second = self.client.post(
            "/v1/chat",
            json={"session_id": "demo", "message": "second"},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["answer"], "seen=1")
        self.assertEqual(second.json()["answer"], "seen=3")
        self.assertEqual(second.json()["retained_messages"], 4)

    def test_reset_discards_previous_history(self) -> None:
        self.client.post(
            "/v1/chat",
            json={"session_id": "demo", "message": "first"},
        )
        response = self.client.post(
            "/v1/chat",
            json={"session_id": "demo", "message": "again", "reset": True},
        )

        self.assertEqual(response.json()["answer"], "seen=1")

    def test_rejects_invalid_session_id(self) -> None:
        response = self.client.post(
            "/v1/chat",
            json={"session_id": "bad id", "message": "hello"},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
