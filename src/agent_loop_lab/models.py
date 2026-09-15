"""Small data contracts shared by the agent, model adapter, and tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


Role = Literal["system", "developer", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolError:
    code: str
    message: str
    retryable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    name: str | None = None
    tool_call: ToolCall | None = None
    call_id: str | None = None
    tool_result: ToolResult | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    name: str
    ok: bool
    content: str
    attempts: int = 1
    duration_ms: float = 0.0
    data: Any | None = None
    error: ToolError | None = None
    truncated: bool = False

    def to_observation(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "ok": self.ok,
            "data": self.data if self.ok else None,
            "error": (
                {
                    "code": self.error.code,
                    "message": self.error.message,
                    "retryable": self.error.retryable,
                    "details": dict(self.error.details),
                }
                if self.error is not None
                else None
            ),
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
        }


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
    def call(
        cls,
        name: str,
        arguments: Mapping[str, Any],
        *,
        call_id: str | None = None,
    ) -> "ModelResponse":
        return cls(
            tool_call=ToolCall(
                name=name,
                arguments=arguments,
                call_id=call_id,
            )
        )
