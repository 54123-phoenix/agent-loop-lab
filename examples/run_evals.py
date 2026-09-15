from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_loop_lab import Agent, EvaluationCase, ModelResponse, build_default_registry
from agent_loop_lab.evaluation import load_jsonl, run_evaluation
from agent_loop_lab.mock_model import ScriptedModel


def make_agent(case: EvaluationCase) -> Agent:
    scripts = {
        "direct-answer": [ModelResponse.final("done")],
        "calculator": [
            ModelResponse.call("calculator", {"expression": "6 * 7"}),
            ModelResponse.final("The result is 42."),
        ],
        "unknown-tool": [
            ModelResponse.call("missing", {}),
            ModelResponse.final("I could not use that tool."),
        ],
    }
    return Agent(ScriptedModel(scripts[case.case_id]), build_default_registry())


def main() -> None:
    summary = run_evaluation(load_jsonl(ROOT / "evals" / "basic.jsonl"), make_agent)
    for result in summary.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.case_id} steps={result.run.steps}")
        for reason in result.reasons:
            print(f"  {reason}")
    print(
        f"passed={summary.passed}/{summary.total} "
        f"pass_rate={summary.pass_rate:.0%} "
        f"average_steps={summary.average_steps:.2f}"
    )


if __name__ == "__main__":
    main()
