"""Run budgets and explicit approval policies for tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import ToolCall
from .tools import ToolSpec


@dataclass(frozen=True, slots=True)
class RunBudget:
    max_steps: int = 8
    max_tool_calls: int = 16
    max_wall_time_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls cannot be negative")
        if self.max_wall_time_seconds <= 0:
            raise ValueError("max_wall_time_seconds must be positive")


class ToolApprovalPolicy(Protocol):
    def approve(self, call: ToolCall, spec: ToolSpec) -> bool: ...


class SafeToolApprovalPolicy:
    """Run read-only tools automatically and require approval for side effects."""

    def approve(self, call: ToolCall, spec: ToolSpec) -> bool:
        del call
        return spec.side_effect == "none"


class AllowAllToolsPolicy:
    """Explicit opt-in policy for trusted demos and controlled environments."""

    def approve(self, call: ToolCall, spec: ToolSpec) -> bool:
        del call, spec
        return True
