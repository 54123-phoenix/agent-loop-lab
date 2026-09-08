"""Small data contracts shared by the agent, model adapter, and tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


Role = Literal["user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    name: str
    ok: bool
    content: str


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Exactly one of final_answer and tool_call must be present."""

    final_answer: str | None = None
    tool_call: ToolCall | None = None

    def __post_init__(self) -> None:
        has_answer = self.final_answer is not None
        has_call = self.tool_call is not None
        if has_answer == has_call:
            raise ValueError("ModelResponse requires exactly one final answer or tool call")

    @classmethod
    def final(cls, answer: str) -> "ModelResponse":
        return cls(final_answer=answer)

    @classmethod
    def call(cls, name: str, arguments: Mapping[str, Any]) -> "ModelResponse":
        return cls(tool_call=ToolCall(name=name, arguments=arguments))
