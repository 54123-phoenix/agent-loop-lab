from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_loop_lab.models import ToolCall
from agent_loop_lab.tools import build_default_registry


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


if __name__ == "__main__":
    unittest.main()
