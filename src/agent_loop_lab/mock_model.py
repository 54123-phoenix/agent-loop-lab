"""Deterministic model adapters for demos and tests."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

from .models import Message, ModelResponse


class ScriptedModel:
    """Return pre-defined responses so the loop can be tested without an API key."""

    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = deque(responses)

    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> ModelResponse:
        del messages, tools
        if not self._responses:
            raise RuntimeError("ScriptedModel has no responses left")
        return self._responses.popleft()
