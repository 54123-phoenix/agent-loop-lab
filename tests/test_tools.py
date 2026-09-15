from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_loop_lab.models import ToolCall
from agent_loop_lab.tools import ToolSpec, build_default_registry


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_default_registry()

    def test_calculator_handles_arithmetic(self) -> None:
        result = self.registry.execute(
            ToolCall("calculator", {"expression": "(2 + 3) * 4"})
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "20")

    def test_calculator_rejects_function_calls(self) -> None:
        result = self.registry.execute(
            ToolCall("calculator", {"expression": "__import__('os').getcwd()"})
        )
        self.assertFalse(result.ok)
        self.assertIn("ValueError", result.content)

    def test_word_count(self) -> None:
        result = self.registry.execute(ToolCall("word_count", {"text": "one two three"}))
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "3")

    def test_json_schema_rejects_wrong_type(self) -> None:
        result = self.registry.execute(ToolCall("calculator", {"expression": 42}))
        self.assertFalse(result.ok)
        self.assertIn("Invalid arguments", result.content)

    def test_json_schema_rejects_extra_argument(self) -> None:
        result = self.registry.execute(
            ToolCall("word_count", {"text": "one", "unexpected": True})
        )
        self.assertFalse(result.ok)
        self.assertIn("Additional properties", result.content)

    def test_duplicate_tool_registration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.registry.register(
                ToolSpec(
                    "calculator",
                    "duplicate",
                    {"type": "object"},
                    lambda arguments: "unused",
                )
            )

    def test_sync_tool_retries(self) -> None:
        attempts = 0

        def flaky(arguments):
            nonlocal attempts
            del arguments
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary")
            return "recovered"

        registry = build_default_registry()
        registry.register(
            ToolSpec(
                "flaky",
                "Fail once",
                {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                flaky,
                max_attempts=2,
            )
        )

        result = registry.execute(ToolCall("flaky", {}))

        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)


if __name__ == "__main__":
    unittest.main()
