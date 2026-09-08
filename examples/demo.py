from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_loop_lab import Agent, ModelResponse, build_default_registry
from agent_loop_lab.mock_model import ScriptedModel


def main() -> None:
    model = ScriptedModel(
        [
            ModelResponse.call("calculator", {"expression": "6 * 7"}),
            ModelResponse.final("6 × 7 = 42"),
        ]
    )
    run = Agent(model, build_default_registry()).run("请计算 6 × 7")
    for message in run.messages:
        print(f"[{message.role}] {message.content}")
    print(f"stop_reason={run.stop_reason}, steps={run.steps}")


if __name__ == "__main__":
    main()
