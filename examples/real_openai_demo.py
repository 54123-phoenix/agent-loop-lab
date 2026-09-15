from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_loop_lab import Agent, build_default_registry
from agent_loop_lab.openai_model import OpenAIResponsesModel


def main() -> None:
    model = OpenAIResponsesModel.from_env()
    run = Agent(model, build_default_registry()).run("请计算 (12 + 8) * 3")
    print(run.answer)
    print(f"stop_reason={run.stop_reason}, steps={run.steps}, run_id={run.run_id}")


if __name__ == "__main__":
    main()
