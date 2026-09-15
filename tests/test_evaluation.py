from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_loop_lab import Agent, EvaluationCase, ModelResponse, build_default_registry
from agent_loop_lab.evaluation import evaluate_run, load_jsonl, run_evaluation
from agent_loop_lab.mock_model import ScriptedModel


class EvaluationTests(unittest.TestCase):
    def test_evaluates_answer_stop_reason_and_tools(self) -> None:
        case = EvaluationCase(
            "calculator",
            "calculate",
            "42",
            expected_tools=("calculator",),
        )
        run = Agent(
            ScriptedModel(
                [
                    ModelResponse.call("calculator", {"expression": "6 * 7"}),
                    ModelResponse.final("42"),
                ]
            ),
            build_default_registry(),
        ).run(case.prompt)

        result = evaluate_run(case, run)

        self.assertTrue(result.passed)
        self.assertEqual(result.reasons, ())

    def test_summary_reports_failure_metrics(self) -> None:
        cases = [EvaluationCase("wrong", "hello", "expected")]

        summary = run_evaluation(
            cases,
            lambda case: Agent(
                ScriptedModel([ModelResponse.final("actual")]),
                build_default_registry(),
            ),
        )

        self.assertEqual(summary.total, 1)
        self.assertEqual(summary.passed, 0)
        self.assertEqual(summary.pass_rate, 0.0)
        self.assertIn("answer", summary.results[0].reasons[0])

    def test_load_jsonl_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(
                '{"case_id":"same","prompt":"one"}\n'
                '{"case_id":"same","prompt":"two"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_jsonl(path)


if __name__ == "__main__":
    unittest.main()
