"""A framework-free agent loop with explicit stop conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .models import Message, ModelResponse
from .tools import ToolRegistry


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


class Agent:
    def __init__(
        self,
        model: ModelAdapter,
        tools: ToolRegistry,
        *,
        max_steps: int = 8,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._model = model
        self._tools = tools
        self._max_steps = max_steps

    def run(self, user_input: str) -> AgentRun:
        if not user_input.strip():
            raise ValueError("user_input cannot be empty")

        messages = [Message(role="user", content=user_input.strip())]

        for step in range(1, self._max_steps + 1):
            response = self._model.respond(tuple(messages), self._tools.schemas())

            if response.final_answer is not None:
                messages.append(Message(role="assistant", content=response.final_answer))
                return AgentRun(
                    answer=response.final_answer,
                    messages=tuple(messages),
                    steps=step,
                    stop_reason="final_answer",
                )

            call = response.tool_call
            if call is None:  # Defensive guard; ModelResponse also enforces this.
                raise RuntimeError("Model returned neither an answer nor a tool call")

            messages.append(
                Message(
                    role="assistant",
                    content=f"tool_call:{call.name} arguments={dict(call.arguments)!r}",
                )
            )
            result = self._tools.execute(call)
            messages.append(
                Message(
                    role="tool",
                    name=result.name,
                    content=f"ok={result.ok} content={result.content}",
                )
            )

        return AgentRun(
            answer=None,
            messages=tuple(messages),
            steps=self._max_steps,
            stop_reason="max_steps",
        )
