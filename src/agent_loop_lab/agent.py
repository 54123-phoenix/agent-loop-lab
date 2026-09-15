"""A framework-free agent loop with explicit stop conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence
from uuid import uuid4

from .context import ContextManager
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
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._model = model
        self._tools = tools
        self._max_steps = max_steps
        self._tracer = tracer
        self._context_manager = context_manager

    def run(
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

            call = response.tool_call
            if call is None:  # Defensive guard; ModelResponse also enforces this.
                raise RuntimeError("Model returned neither an answer nor a tool call")

            if call.call_id is None:
                from dataclasses import replace

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
            result = self._tools.execute(call)
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
            answer=None,
            messages=tuple(messages),
            steps=self._max_steps,
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
