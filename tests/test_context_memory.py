from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_loop_lab import (
    ApproximateTokenEstimator,
    ContextBudget,
    ContextManager,
    Message,
    ConversationConflictError,
    SQLiteConversationStore,
    ToolCall,
)


class ContextManagerTests(unittest.TestCase):
    def test_keeps_latest_message_within_budget(self) -> None:
        manager = ContextManager(
            ContextBudget(max_input_tokens=40, reserved_output_tokens=5, summary_tokens=0),
            estimator=ApproximateTokenEstimator(characters_per_token=1),
        )
        messages = [
            Message("user", "old message that should leave the active context"),
            Message("assistant", "old answer"),
            Message("user", "current"),
        ]

        selection = manager.build(messages)

        self.assertEqual(selection.messages[-1].content, "current")
        self.assertGreater(selection.dropped_messages, 0)
        self.assertLessEqual(selection.estimated_input_tokens, 40)

    def test_tool_call_and_output_are_an_atomic_unit(self) -> None:
        call = ToolCall("calculator", {"expression": "1+1"}, "call_1")
        manager = ContextManager(
            ContextBudget(max_input_tokens=70, reserved_output_tokens=5, summary_tokens=0),
            estimator=ApproximateTokenEstimator(characters_per_token=1),
        )
        messages = [
            Message("user", "x" * 60),
            Message("assistant", "tool", tool_call=call, call_id="call_1"),
            Message("tool", "2", "calculator", call_id="call_1"),
        ]

        selection = manager.build(messages)

        self.assertEqual([message.role for message in selection.messages], ["assistant", "tool"])

    def test_summarizes_dropped_messages(self) -> None:
        manager = ContextManager(
            ContextBudget(max_input_tokens=90, reserved_output_tokens=5, summary_tokens=20),
            estimator=ApproximateTokenEstimator(characters_per_token=1),
        )
        messages = [
            Message("user", "old " * 20),
            Message("assistant", "earlier answer"),
            Message("user", "latest"),
        ]

        selection = manager.build(messages)

        self.assertTrue(selection.summary_included)
        self.assertEqual(selection.messages[0].role, "system")


class SQLiteConversationStoreTests(unittest.TestCase):
    def test_history_survives_store_recreation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.db"
            first = SQLiteConversationStore(path)
            first.save("session", [Message("user", "hello")])

            second = SQLiteConversationStore(path)

            self.assertEqual(second.load("session"), (Message("user", "hello"),))

    def test_save_appends_only_new_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteConversationStore(Path(directory) / "history.db")
            first = Message("user", "hello")
            second = Message("assistant", "hi")
            store.save("session", [first])
            store.save("session", [first, second])

            self.assertEqual(store.load("session"), (first, second))

    def test_rejects_rewriting_existing_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteConversationStore(Path(directory) / "history.db")
            store.save("session", [Message("user", "original")])

            with self.assertRaises(ValueError):
                store.save("session", [Message("user", "changed")])

    def test_rejects_stale_session_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteConversationStore(Path(directory) / "history.db")
            snapshot = store.load_snapshot("session")
            store.save_if_version(
                "session",
                [Message("user", "first")],
                expected_version=snapshot.version,
            )

            with self.assertRaises(ConversationConflictError):
                store.save_if_version(
                    "session",
                    [Message("user", "stale")],
                    expected_version=snapshot.version,
                )


if __name__ == "__main__":
    unittest.main()
