from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_loop_lab import Agent, ModelResponse, build_default_registry
from agent_loop_lab.mock_model import ScriptedModel


class AgentTests(unittest.TestCase):
    def test_returns_a_direct_final_answer(self) -> None:
        model = ScriptedModel([ModelResponse.final("done")])
        run = Agent(model, build_default_registry()).run("hello")

        self.assertEqual(run.answer, "done")
        self.assertEqual(run.stop_reason, "final_answer")
        self.assertEqual(run.steps, 1)

    def test_executes_a_tool_and_adds_observation(self) -> None:
        model = ScriptedModel(
            [
                ModelResponse.call("calculator", {"expression": "6 * 7"}),
                ModelResponse.final("The result is 42."),
            ]
        )
        run = Agent(model, build_default_registry()).run("Calculate 6 * 7")

        self.assertEqual(run.answer, "The result is 42.")
        self.assertEqual(run.steps, 2)
        self.assertIn("content=42", run.messages[2].content)

    def test_unknown_tool_becomes_a_failed_observation(self) -> None:
        model = ScriptedModel(
            [
                ModelResponse.call("missing", {}),
                ModelResponse.final("I could not use that tool."),
            ]
        )
        run = Agent(model, build_default_registry()).run("Use a missing tool")

        self.assertIn("ok=False", run.messages[2].content)
        self.assertIn("Unknown tool", run.messages[2].content)

    def test_missing_argument_becomes_a_failed_observation(self) -> None:
        model = ScriptedModel(
            [
                ModelResponse.call("calculator", {}),
                ModelResponse.final("The expression is missing."),
            ]
        )
        run = Agent(model, build_default_registry()).run("Calculate")

        self.assertIn("Missing required arguments", run.messages[2].content)

    def test_stops_at_max_steps(self) -> None:
        model = ScriptedModel(
            [
                ModelResponse.call("word_count", {"text": "one two"}),
                ModelResponse.call("word_count", {"text": "three four"}),
            ]
        )
        run = Agent(model, build_default_registry(), max_steps=2).run("Keep calling")

        self.assertIsNone(run.answer)
        self.assertEqual(run.stop_reason, "max_steps")
        self.assertEqual(run.steps, 2)

    def test_rejects_empty_input(self) -> None:
        model = ScriptedModel([ModelResponse.final("unused")])
        with self.assertRaises(ValueError):
            Agent(model, build_default_registry()).run("   ")


class ModelResponseTests(unittest.TestCase):
    def test_rejects_neither_answer_nor_call(self) -> None:
        with self.assertRaises(ValueError):
            ModelResponse()

    def test_rejects_answer_and_call_together(self) -> None:
        with self.assertRaises(ValueError):
            ModelResponse(
                final_answer="done",
                tool_call=ModelResponse.call("calculator", {}).tool_call,
            )


if __name__ == "__main__":
    unittest.main()
