"""Async agent loop with model/tool timeouts and the same message contract."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from time import perf_counter
from typing import Protocol, Sequence
from uuid import uuid4

from .agent import AgentRun
from .context import ContextManager
from .guardrails import RunBudget, SafeToolApprovalPolicy, ToolApprovalPolicy
from .models import Message, ModelResponse
from .tools import ToolRegistry
from .tracing import TraceEvent, TraceSink


class AsyncModelAdapter(Protocol):
    async def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> ModelResponse: ...


class AsyncAgent:
    def __init__(
        self,
        model: AsyncModelAdapter,
        tools: ToolRegistry,
        *,
        max_steps: int = 8,
        model_timeout_seconds: float = 30.0,
        tracer: TraceSink | None = None,
        context_manager: ContextManager | None = None,
        budget: RunBudget | None = None,
        approval_policy: ToolApprovalPolicy | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if model_timeout_seconds <= 0:
            raise ValueError("model_timeout_seconds must be positive")
        self._model = model
        self._tools = tools
        self._budget = budget or RunBudget(max_steps=max_steps)
        self._model_timeout_seconds = model_timeout_seconds
        self._tracer = tracer
        self._context_manager = context_manager
        self._approval_policy = approval_policy or SafeToolApprovalPolicy()

    async def run(
        self,
        user_input: str,
        *,
        history: Sequence[Message] = (),
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
                self._context_manager.build(tuple(messages), schemas)
                if self._context_manager is not None
                else None
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
                },
            )
            try:
                response = await asyncio.wait_for(
                    self._model.respond(model_messages, schemas),
                    timeout=self._model_timeout_seconds,
                )
            except TimeoutError:
                error = f"Model timed out after {self._model_timeout_seconds:g}s"
                self._emit(run_id, "run_finished", step, {"reason": "model_timeout"})
                return AgentRun(
                    None,
                    tuple(messages),
                    step,
                    "model_timeout",
                    run_id,
                    error,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self._emit(run_id, "run_finished", step, {"reason": "model_error"})
                return AgentRun(
                    None,
                    tuple(messages),
                    step,
                    "model_error",
                    run_id,
                    error,
                )

            if response.final_answer is not None:
                messages.append(Message(role="assistant", content=response.final_answer))
                self._emit(run_id, "run_finished", step, {"reason": "final_answer"})
                return AgentRun(
                    response.final_answer,
                    tuple(messages),
                    step,
                    "final_answer",
                    run_id,
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

            prepared_calls = []
            approved_calls = []
            approval_results = {}
            for index, original_call in enumerate(calls, start=1):
                call = original_call
                if call.call_id is None:
                    call = replace(call, call_id=f"call_{run_id}_{step}_{index}")
                prepared_calls.append(call)
                spec = self._tools.get(call.name)
                approved = spec is None or self._approval_policy.approve(call, spec)
                self._emit(
                    run_id,
                    "tool_started",
                    step,
                    {"tool": call.name, "approved": approved},
                )
                if approved:
                    approved_calls.append(call)
                else:
                    approval_results[call.call_id] = self._tools.approval_required(call.name)

            executed = iter(await self._tools.execute_many_async(approved_calls))
            for call in prepared_calls:
                result = approval_results.get(call.call_id) or next(executed)
                messages.append(
                    Message(
                        role="assistant",
                        content=f"tool_call:{call.name} arguments={dict(call.arguments)!r}",
                        tool_call=call,
                        call_id=call.call_id,
                    )
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
            None,
            tuple(messages),
            self._budget.max_steps,
            "max_steps",
            run_id,
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
