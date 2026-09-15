"""Tool contracts, validation, and a small safe calculator."""

from __future__ import annotations

import ast
import asyncio
import inspect
import operator
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Any, Awaitable, Callable, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .models import ToolCall, ToolResult


ToolHandler = Callable[[Mapping[str, Any]], str | Awaitable[str]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: ToolHandler
    timeout_seconds: float = 5.0
    max_attempts: int = 1
    retry_backoff_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Tool name cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        try:
            Draft202012Validator.check_schema(dict(self.parameters))
        except SchemaError as exc:
            raise ValueError(f"Invalid tool schema: {exc.message}") from exc

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
            "strict": True,
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
        prepared = self._prepare(call)
        if isinstance(prepared, ToolResult):
            return prepared

        started = perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, prepared.max_attempts + 1):
            try:
                if inspect.iscoroutinefunction(prepared.handler):
                    raise TypeError("Async tool handlers require execute_async()")
                content = prepared.handler(call.arguments)
                if inspect.isawaitable(content):
                    close = getattr(content, "close", None)
                    if close is not None:
                        close()
                    raise TypeError("Async tool handlers require execute_async()")
                return ToolResult(
                    call.name,
                    True,
                    str(content),
                    attempts=attempt,
                    duration_ms=(perf_counter() - started) * 1000,
                )
            except Exception as exc:  # Tool failures become observations for the model.
                last_error = exc
                if (
                    attempt < prepared.max_attempts
                    and prepared.retry_backoff_seconds
                ):
                    sleep(prepared.retry_backoff_seconds * (2 ** (attempt - 1)))

        assert last_error is not None
        return ToolResult(
            call.name,
            False,
            f"{type(last_error).__name__}: {last_error}",
            attempts=prepared.max_attempts,
            duration_ms=(perf_counter() - started) * 1000,
        )

    async def execute_async(self, call: ToolCall) -> ToolResult:
        prepared = self._prepare(call)
        if isinstance(prepared, ToolResult):
            return prepared

        started = perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, prepared.max_attempts + 1):
            try:
                content = await asyncio.wait_for(
                    self._invoke_async(prepared, call),
                    timeout=prepared.timeout_seconds,
                )
                return ToolResult(
                    call.name,
                    True,
                    str(content),
                    attempts=attempt,
                    duration_ms=(perf_counter() - started) * 1000,
                )
            except Exception as exc:
                last_error = exc
                if (
                    attempt < prepared.max_attempts
                    and prepared.retry_backoff_seconds
                ):
                    await asyncio.sleep(
                        prepared.retry_backoff_seconds * (2 ** (attempt - 1))
                    )

        assert last_error is not None
        return ToolResult(
            call.name,
            False,
            f"{type(last_error).__name__}: {last_error}",
            attempts=prepared.max_attempts,
            duration_ms=(perf_counter() - started) * 1000,
        )

    def _prepare(self, call: ToolCall) -> ToolSpec | ToolResult:
        spec = self._tools.get(call.name)
        if spec is None:
            return ToolResult(call.name, False, f"Unknown tool: {call.name}")
        try:
            Draft202012Validator(spec.parameters).validate(dict(call.arguments))
        except ValidationError as exc:
            if exc.validator == "required":
                missing = [
                    key
                    for key in spec.parameters.get("required", ())
                    if key not in call.arguments
                ]
                return ToolResult(
                    call.name,
                    False,
                    "Missing required arguments: " + ", ".join(missing),
                )
            return ToolResult(call.name, False, f"Invalid arguments: {exc.message}")
        return spec

    @staticmethod
    async def _invoke_async(spec: ToolSpec, call: ToolCall) -> str:
        if inspect.iscoroutinefunction(spec.handler):
            return await spec.handler(call.arguments)
        content = await asyncio.to_thread(spec.handler, call.arguments)
        if inspect.isawaitable(content):
            return await content
        return content


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
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
            handler=calculate,
        )
    )
    registry.register(
        ToolSpec(
            name="word_count",
            description="Count whitespace-separated words in text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=word_count,
        )
    )
    return registry
