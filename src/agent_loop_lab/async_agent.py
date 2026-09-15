"""Async agent loop with model/tool timeouts and the same message contract."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Protocol, Sequence
from uuid import uuid4

from .agent import AgentRun
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
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if model_timeout_seconds <= 0:
            raise ValueError("model_timeout_seconds must be positive")
        self._model = model
        self._tools = tools
        self._max_steps = max_steps
        self._model_timeout_seconds = model_timeout_seconds
        self._tracer = tracer

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

        for step in range(1, self._max_steps + 1):
            self._emit(run_id, "model_requested", step, {"messages": len(messages)})
            try:
                response = await asyncio.wait_for(
                    self._model.respond(tuple(messages), self._tools.schemas()),
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

            call = response.tool_call
            if call is None:
                raise RuntimeError("Model returned neither an answer nor a tool call")
            if call.call_id is None:
                call = replace(call, call_id=f"call_{run_id}_{step}")

            messages.append(
                Message(
                    role="assistant",
                    content=f"tool_call:{call.name} arguments={dict(call.arguments)!r}",
                    tool_call=call,
                    call_id=call.call_id,
                )
            )
            self._emit(run_id, "tool_started", step, {"tool": call.name})
            result = await self._tools.execute_async(call)
            messages.append(
                Message(
                    role="tool",
                    name=result.name,
                    content=f"ok={result.ok} content={result.content}",
                    call_id=call.call_id,
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

        self._emit(run_id, "run_finished", self._max_steps, {"reason": "max_steps"})
        return AgentRun(
            None,
            tuple(messages),
            self._max_steps,
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

