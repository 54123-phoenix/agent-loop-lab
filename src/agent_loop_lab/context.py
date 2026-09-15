"""Token-budgeted context selection that never mutates source history."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from math import ceil
from typing import Protocol, Sequence

from .models import Message


class TokenEstimator(Protocol):
    def text_tokens(self, text: str) -> int: ...


class ApproximateTokenEstimator:
    """Dependency-free estimate; replace with a provider tokenizer when available."""

    def __init__(self, *, characters_per_token: float = 4.0) -> None:
        if characters_per_token <= 0:
            raise ValueError("characters_per_token must be positive")
        self._characters_per_token = characters_per_token

    def text_tokens(self, text: str) -> int:
        return max(1, ceil(len(text) / self._characters_per_token))


class Summarizer(Protocol):
    def summarize(self, messages: Sequence[Message], *, max_tokens: int) -> str: ...


class ExtractiveSummarizer:
    """Predictable local summary for the scaffold; it makes no model request."""

    def __init__(self, estimator: TokenEstimator | None = None) -> None:
        self._estimator = estimator or ApproximateTokenEstimator()

    def summarize(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        if max_tokens < 1:
            return ""
        pieces: list[str] = []
        for message in messages:
            label = message.name or message.role
            content = " ".join(message.content.split())
            pieces.append(f"{label}: {content}")
        summary = " | ".join(pieces)
        max_chars = max_tokens * 4
        if len(summary) > max_chars:
            summary = summary[: max(0, max_chars - 1)].rstrip() + "…"
        return summary


@dataclass(frozen=True, slots=True)
class ContextBudget:
    max_input_tokens: int = 16_000
    reserved_output_tokens: int = 2_000
    summary_tokens: int = 512

    def __post_init__(self) -> None:
        if self.max_input_tokens < 1:
            raise ValueError("max_input_tokens must be positive")
        if not 0 <= self.reserved_output_tokens < self.max_input_tokens:
            raise ValueError("reserved_output_tokens must be within the input budget")
        if self.summary_tokens < 0:
            raise ValueError("summary_tokens cannot be negative")


@dataclass(frozen=True, slots=True)
class ContextSelection:
    messages: tuple[Message, ...]
    estimated_input_tokens: int
    dropped_messages: int
    summary_included: bool


class ContextManager:
    def __init__(
        self,
        budget: ContextBudget | None = None,
        *,
        estimator: TokenEstimator | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        self._budget = budget or ContextBudget()
        self._estimator = estimator or ApproximateTokenEstimator()
        self._summarizer = summarizer or ExtractiveSummarizer(self._estimator)

    def build(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[dict[str, object]] = (),
    ) -> ContextSelection:
        if not messages:
            return ContextSelection((), self._tool_tokens(tool_schemas), 0, False)

        tool_tokens = self._tool_tokens(tool_schemas)
        message_budget = max(
            1,
            self._budget.max_input_tokens
            - self._budget.reserved_output_tokens
            - tool_tokens,
        )
        units = self._atomic_units(messages)
        selected_reversed: list[tuple[Message, ...]] = []
        selected_tokens = 0

        for unit in reversed(units):
            unit_tokens = self._messages_tokens(unit)
            if selected_reversed and selected_tokens + unit_tokens > message_budget:
                break
            if not selected_reversed and unit_tokens > message_budget:
                unit = self._clip_unit(unit, message_budget)
                unit_tokens = self._messages_tokens(unit)
            selected_reversed.append(unit)
            selected_tokens += unit_tokens

        selected_units = list(reversed(selected_reversed))
        selected_count = sum(len(unit) for unit in selected_units)
        dropped = tuple(messages[: len(messages) - selected_count])

        summary_message: Message | None = None
        if dropped and self._budget.summary_tokens:
            summary_room = message_budget - selected_tokens
            summary_prefix = "Earlier conversation summary: "
            summary_overhead = 4 + self._estimator.text_tokens(summary_prefix)
            if summary_room <= summary_overhead and len(selected_units) > 1:
                moved = selected_units.pop(0)
                dropped = dropped + moved
                selected_tokens -= self._messages_tokens(moved)
                summary_room = message_budget - selected_tokens
            summary_limit = min(
                self._budget.summary_tokens,
                max(0, summary_room - summary_overhead),
            )
            summary = self._summarizer.summarize(dropped, max_tokens=summary_limit)
            if summary:
                summary_message = Message(
                    role="system",
                    content=summary_prefix + summary,
                )
                selected_tokens += self._message_tokens(summary_message)

        selected = tuple(message for unit in selected_units for message in unit)
        if summary_message is not None:
            selected = (summary_message,) + selected
        return ContextSelection(
            messages=selected,
            estimated_input_tokens=tool_tokens + selected_tokens,
            dropped_messages=len(dropped),
            summary_included=summary_message is not None,
        )

    def _tool_tokens(self, schemas: Sequence[dict[str, object]]) -> int:
        return self._estimator.text_tokens(
            json.dumps(list(schemas), ensure_ascii=False, sort_keys=True)
        )

    def _message_tokens(self, message: Message) -> int:
        return 4 + self._estimator.text_tokens(message.content)

    def _messages_tokens(self, messages: Sequence[Message]) -> int:
        return sum(self._message_tokens(message) for message in messages)

    @staticmethod
    def _atomic_units(messages: Sequence[Message]) -> list[tuple[Message, ...]]:
        units: list[tuple[Message, ...]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if (
                message.tool_call is not None
                and index + 1 < len(messages)
                and messages[index + 1].role == "tool"
                and messages[index + 1].call_id == message.tool_call.call_id
            ):
                units.append((message, messages[index + 1]))
                index += 2
            else:
                units.append((message,))
                index += 1
        return units

    def _clip_unit(
        self,
        unit: tuple[Message, ...],
        token_budget: int,
    ) -> tuple[Message, ...]:
        per_message = max(1, token_budget // len(unit) - 4)
        max_chars = per_message * 4
        return tuple(
            replace(
                message,
                content=(
                    message.content
                    if len(message.content) <= max_chars
                    else message.content[: max(0, max_chars - 1)].rstrip() + "…"
                ),
            )
            for message in unit
        )
