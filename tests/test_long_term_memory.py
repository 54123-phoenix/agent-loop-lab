from __future__ import annotations

from datetime import datetime, timedelta, timezone
import tempfile
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from agent_loop_lab import (
    Agent,
    ApproximateTokenEstimator,
    ContextBudget,
    ContextManager,
    InMemoryLongTermMemoryStore,
    MemoryRecord,
    Message,
    ModelResponse,
    SQLiteLongTermMemoryStore,
    build_default_registry,
)
from agent_loop_lab.api import create_app


class MemoryAwareModel:
    def respond(self, messages, tools):
        del tools
        memory = next(
            (message.content for message in messages if "long-term memory" in message.content),
            "none",
        )
        return ModelResponse.final(memory)


class LongTermMemoryStoreTests(unittest.TestCase):
    def test_sqlite_memory_survives_recreation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.db"
            created = SQLiteLongTermMemoryStore(path).remember(
                MemoryRecord("session", "preference", "Prefers concise answers")
            )

            loaded = SQLiteLongTermMemoryStore(path).list("session")

            self.assertEqual(loaded, (created,))

    def test_search_is_owner_scoped_and_excludes_expired_records(self) -> None:
        store = InMemoryLongTermMemoryStore()
        store.remember(MemoryRecord("one", "goal", "Learn Python agents"))
        store.remember(MemoryRecord("two", "goal", "Learn Python agents"))
        store.remember(
            MemoryRecord(
                "one",
                "goal",
                "Old Python goal",
                expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
            )
        )

        results = store.search("one", "Python", limit=5)

        self.assertEqual([record.content for record in results], ["Learn Python agents"])

    def test_same_memory_id_can_be_used_by_different_owners(self) -> None:
        store = InMemoryLongTermMemoryStore()
        store.remember(MemoryRecord("one", "fact", "one", memory_id="shared"))
        store.remember(MemoryRecord("two", "fact", "two", memory_id="shared"))

        self.assertTrue(store.forget("one", "shared"))
        self.assertEqual(store.list("two")[0].content, "two")


class MemoryContextTests(unittest.TestCase):
    def test_relevant_memory_is_injected_without_changing_source_history(self) -> None:
        manager = ContextManager(
            ContextBudget(
                max_input_tokens=150,
                reserved_output_tokens=10,
                summary_tokens=0,
                memory_tokens=100,
            ),
            estimator=ApproximateTokenEstimator(characters_per_token=1),
        )
        messages = [Message("user", "What format should you use?")]

        selection = manager.build(messages, memories=["[preference] concise answers"])

        self.assertIn("concise answers", selection.messages[0].content)
        self.assertEqual(selection.retrieved_memories, 1)
        self.assertEqual(messages, [Message("user", "What format should you use?")])
        self.assertLessEqual(selection.estimated_input_tokens, 150)


class MemoryApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memories = InMemoryLongTermMemoryStore()
        app = create_app(
            lambda: Agent(
                MemoryAwareModel(),
                build_default_registry(),
                context_manager=ContextManager(),
            ),
            memory_store=self.memories,
        )
        self.client = TestClient(app)

    def test_memory_can_be_created_recalled_listed_and_deleted(self) -> None:
        created = self.client.post(
            "/v1/sessions/demo/memories",
            json={
                "memory_id": "preference-1",
                "category": "preference",
                "content": "Prefers concise answers",
                "source_event_ids": ["event-1"],
                "confidence": 0.9,
            },
        )
        chat = self.client.post(
            "/v1/chat",
            json={"session_id": "demo", "message": "What answers do I prefer?"},
        )
        listed = self.client.get("/v1/sessions/demo/memories")
        deleted = self.client.delete("/v1/sessions/demo/memories/preference-1")

        self.assertEqual(created.status_code, 201)
        self.assertIn("Prefers concise answers", chat.json()["answer"])
        self.assertEqual(len(listed.json()["memories"]), 1)
        self.assertTrue(deleted.json()["deleted"])
        self.assertEqual(self.client.get("/v1/sessions/demo/memories").json()["memories"], [])


if __name__ == "__main__":
    unittest.main()
