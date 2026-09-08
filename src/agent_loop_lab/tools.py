"""Tool contracts, validation, and a small safe calculator."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .models import ToolCall, ToolResult


ToolHandler = Callable[[Mapping[str, Any]], str]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    required: tuple[str, ...]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "required": list(self.required),
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in sorted(self._tools)]

    def execute(self, call: ToolCall) -> ToolResult:
        spec = self._tools.get(call.name)
        if spec is None:
            return ToolResult(call.name, False, f"Unknown tool: {call.name}")

        missing = [key for key in spec.required if key not in call.arguments]
        if missing:
            return ToolResult(
                call.name,
                False,
                "Missing required arguments: " + ", ".join(missing),
            )

        try:
            return ToolResult(call.name, True, str(spec.handler(call.arguments)))
        except Exception as exc:  # Tool failures become observations for the model.
            return ToolResult(call.name, False, f"{type(exc).__name__}: {exc}")


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent is limited to 10")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand))
    raise ValueError("Only numeric arithmetic expressions are allowed")


def calculate(arguments: Mapping[str, Any]) -> str:
    expression = str(arguments["expression"]).strip()
    if not expression or len(expression) > 120:
        raise ValueError("Expression must contain 1 to 120 characters")
    tree = ast.parse(expression, mode="eval")
    value = _evaluate_node(tree.body)
    return str(int(value)) if value.is_integer() else str(value)


def word_count(arguments: Mapping[str, Any]) -> str:
    text = str(arguments["text"])
    return str(len(text.split()))


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="calculator",
            description="Evaluate a numeric arithmetic expression.",
            required=("expression",),
            handler=calculate,
        )
    )
    registry.register(
        ToolSpec(
            name="word_count",
            description="Count whitespace-separated words in text.",
            required=("text",),
            handler=word_count,
        )
    )
    return registry
