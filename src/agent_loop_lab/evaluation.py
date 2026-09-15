"""Versioned evaluation cases and small deterministic metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable, Iterable

from .agent import Agent, AgentRun


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    prompt: str
    expected_answer: str | None = None
    expected_stop_reason: str = "final_answer"
    expected_tools: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    case_id: str
    passed: bool
    run: AgentRun
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    total: int
    passed: int
    pass_rate: float
    average_steps: float
    results: tuple[EvaluationResult, ...]


def load_jsonl(path: str | Path) -> tuple[EvaluationCase, ...]:
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                case = EvaluationCase(
                    case_id=str(payload["case_id"]),
                    prompt=str(payload["prompt"]),
                    expected_answer=payload.get("expected_answer"),
                    expected_stop_reason=str(
                        payload.get("expected_stop_reason", "final_answer")
                    ),
                    expected_tools=tuple(payload.get("expected_tools", ())),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid evaluation case on line {line_number}") from exc
            if not case.case_id or not case.prompt.strip():
                raise ValueError(f"Empty case_id or prompt on line {line_number}")
            if case.case_id in seen_ids:
                raise ValueError(f"Duplicate evaluation case_id: {case.case_id}")
            seen_ids.add(case.case_id)
            cases.append(case)
    return tuple(cases)


def evaluate_run(case: EvaluationCase, run: AgentRun) -> EvaluationResult:
    reasons: list[str] = []
    if run.stop_reason != case.expected_stop_reason:
        reasons.append(
            f"stop_reason: expected {case.expected_stop_reason!r}, got {run.stop_reason!r}"
        )
    if case.expected_answer is not None and run.answer != case.expected_answer:
        reasons.append(
            f"answer: expected {case.expected_answer!r}, got {run.answer!r}"
        )
    used_tools = tuple(
        message.tool_call.name
        for message in run.messages
        if message.tool_call is not None
    )
    if used_tools != case.expected_tools:
        reasons.append(f"tools: expected {case.expected_tools!r}, got {used_tools!r}")
    return EvaluationResult(case.case_id, not reasons, run, tuple(reasons))


def run_evaluation(
    cases: Iterable[EvaluationCase],
    agent_factory: Callable[[EvaluationCase], Agent],
) -> EvaluationSummary:
    results = tuple(
        evaluate_run(case, agent_factory(case).run(case.prompt)) for case in cases
    )
    total = len(results)
    passed = sum(result.passed for result in results)
    return EvaluationSummary(
        total=total,
        passed=passed,
        pass_rate=(passed / total if total else 0.0),
        average_steps=(fmean(result.run.steps for result in results) if total else 0.0),
        results=results,
    )

