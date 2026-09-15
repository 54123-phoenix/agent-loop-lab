"""A framework-free agent loop with explicit stop conditions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Protocol, Sequence
from uuid import uuid4

from .context import ContextManager
from .guardrails import RunBudget, SafeToolApprovalPolicy, ToolApprovalPolicy
from .models import Message, ModelResponse
from .tools import ToolRegistry
from .tracing import TraceEvent, TraceSink


class ModelAdapter(Protocol):
    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> ModelResponse: ...


@dataclass(frozen=True, slots=True)
class AgentRun:
    answer: str | None
    messages: tuple[Message, ...]
    steps: int
    stop_reason: str
    run_id: str = ""
    error: str | None = None


class Agent:
    def __init__(
        self,
        model: ModelAdapter,
        tools: ToolRegistry,
        *,
        max_steps: int = 8,
        tracer: TraceSink | None = None,
        context_manager: ContextManager | None = None,
        budget: RunBudget | None = None,
        approval_policy: ToolApprovalPolicy | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._model = model
        self._tools = tools
        self._budget = budget or RunBudget(max_steps=max_steps)
        self._tracer = tracer
        self._context_manager = context_manager
        self._approval_policy = approval_policy or SafeToolApprovalPolicy()

    def run(
        self,
        user_input: str,
        *,
        history: Sequence[Message] = (),
        memories: Sequence[str] = (),
    ) -> AgentRun:
        if not user_input.strip():
            raise ValueError("user_input cannot be empty")

        run_id = uuid4().hex
        messages = list(history)
        messages.append(Message(role="user", content=user_input.strip()))
        self._emit(run_id, "run_started", 0, {"history_messages": len(history)})
        started = perf_counter()
        tool_calls_used = 0

        for step in range(1, self._budget.max_steps + 1):
            if perf_counter() - started >= self._budget.max_wall_time_seconds:
                self._emit(run_id, "run_finished", step - 1, {"reason": "time_budget"})
                return AgentRun(
                    None,
                    tuple(messages),
                    step - 1,
                    "time_budget",
                    run_id,
                    "Run exceeded its wall-time budget",
                )
            schemas = self._tools.schemas()
            selection = (
                self._context_manager.build(tuple(messages), schemas, memories)
                if self._context_manager is not None
                else (
                    ContextManager().build(tuple(messages), schemas, memories)
                    if memories
                    else None
                )
            )
            model_messages = selection.messages if selection is not None else tuple(messages)
            self._emit(
                run_id,
                "model_requested",
                step,
                {
                    "messages": len(model_messages),
                    "dropped_messages": selection.dropped_messages if selection else 0,
                    "estimated_input_tokens": (
                        selection.estimated_input_tokens if selection else None
                    ),
                    "retrieved_memories": (
                        selection.retrieved_memories if selection else 0
                    ),
                },
            )
            try:
                response = self._model.respond(model_messages, schemas)
            except Exception as exc:
                self._emit(
                    run_id,
                    "run_finished",
                    step,
                    {"reason": "model_error", "error_type": type(exc).__name__},
                )
                raise

            if response.final_answer is not None:
                messages.append(Message(role="assistant", content=response.final_answer))
                self._emit(run_id, "run_finished", step, {"reason": "final_answer"})
                return AgentRun(
                    answer=response.final_answer,
                    messages=tuple(messages),
                    steps=step,
                    stop_reason="final_answer",
                    run_id=run_id,
                )

            calls = response.requested_tool_calls()
            if not calls:
                raise RuntimeError("Model returned neither an answer nor a tool call")
            if tool_calls_used + len(calls) > self._budget.max_tool_calls:
                self._emit(run_id, "run_finished", step, {"reason": "max_tool_calls"})
                return AgentRun(
                    None,
                    tuple(messages),
                    step,
                    "max_tool_calls",
                    run_id,
                    "Run exceeded its tool-call budget",
                )
            tool_calls_used += len(calls)

            for index, original_call in enumerate(calls, start=1):
                call = original_call
                if call.call_id is None:
                    call = replace(call, call_id=f"call_{run_id}_{step}_{index}")
                messages.append(
                    Message(
                        role="assistant",
                        content=f"tool_call:{call.name} arguments={dict(call.arguments)!r}",
                        tool_call=call,
                        call_id=call.call_id,
                    )
                )
                spec = self._tools.get(call.name)
                approved = spec is None or self._approval_policy.approve(call, spec)
                self._emit(
                    run_id,
                    "tool_started",
                    step,
                    {"tool": call.name, "approved": approved},
                )
                result = (
                    self._tools.execute(call)
                    if approved
                    else self._tools.approval_required(call.name)
                )
                messages.append(
                    Message(
                        role="tool",
                        name=result.name,
                        content=f"ok={result.ok} content={result.content}",
                        call_id=call.call_id,
                        tool_result=result,
                    )
                )
                self._emit(
                    run_id,
                    "tool_finished",
                    step,
                    {
                        "tool": result.name,
                        "ok": result.ok,
                        "attempts": result.attempts,
                        "duration_ms": result.duration_ms,
                    },
                )

        self._emit(
            run_id,
            "run_finished",
            self._budget.max_steps,
            {"reason": "max_steps"},
        )
        return AgentRun(
            answer=None,
            messages=tuple(messages),
            steps=self._budget.max_steps,
            stop_reason="max_steps",
            run_id=run_id,
        )

    def _emit(
        self,
        run_id: str,
        kind: str,
        step: int,
        details: dict[str, object],
    ) -> None:
        if self._tracer is not None:
            self._tracer.emit(TraceEvent(run_id, kind, step, details))
