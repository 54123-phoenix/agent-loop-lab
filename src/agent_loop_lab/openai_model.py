"""OpenAI Responses API adapter for the framework-neutral agent loop."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

from .models import Message, ModelResponse


class OpenAIResponsesModel:
    """Translate local messages and tool schemas to the OpenAI Responses API."""

    def __init__(
        self,
        model: str,
        *,
        instructions: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - environment dependent.
                raise RuntimeError(
                    "Install the OpenAI extra: pip install -e '.[openai]'"
                ) from exc
            client = OpenAI()
        self._client = client
        self._model = model
        self._instructions = instructions

    @classmethod
    def from_env(cls, *, client: Any | None = None) -> "OpenAIResponsesModel":
        model = os.getenv("OPENAI_MODEL", "").strip()
        if not model:
            raise RuntimeError("OPENAI_MODEL is required")
        return cls(
            model,
            instructions=os.getenv("AGENT_INSTRUCTIONS") or None,
            client=client,
        )

    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> ModelResponse:
        request: dict[str, Any] = {
            "model": self._model,
            "input": self._to_input(messages),
            "tools": list(tools),
            "parallel_tool_calls": True,
        }
        if self._instructions:
            request["instructions"] = self._instructions

        response = self._client.responses.create(**request)
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: Any) -> ModelResponse:
        calls = []
        for item in getattr(response, "output", ()):
            if OpenAIResponsesModel._get(item, "type") != "function_call":
                continue
            name = OpenAIResponsesModel._get(item, "name")
            call_id = OpenAIResponsesModel._get(item, "call_id")
            if not isinstance(name, str) or not name:
                raise RuntimeError("Model function call is missing a name")
            if not isinstance(call_id, str) or not call_id:
                raise RuntimeError("Model function call is missing a call_id")
            raw_arguments = OpenAIResponsesModel._get(item, "arguments")
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError("Model returned invalid tool arguments JSON") from exc
            if not isinstance(arguments, dict):
                raise RuntimeError("Model tool arguments must decode to an object")
            calls.append(ModelResponse.call(name, arguments, call_id=call_id).tool_call)

        if len(calls) == 1:
            return ModelResponse(tool_call=calls[0])
        if calls:
            return ModelResponse.call_many(calls)

        output_text = str(getattr(response, "output_text", "")).strip()
        if not output_text:
            raise RuntimeError("Model returned neither text nor a function call")
        return ModelResponse.final(output_text)

    @staticmethod
    def _to_input(messages: Sequence[Message]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                if not message.call_id:
                    raise ValueError("Tool messages require call_id for the Responses API")
                output = (
                    json.dumps(
                        message.tool_result.to_observation(),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if message.tool_result is not None
                    else message.content
                )
                result.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.call_id,
                        "output": output,
                    }
                )
            elif message.tool_call is not None:
                if not message.tool_call.call_id:
                    raise ValueError("Tool calls require call_id for the Responses API")
                result.append(
                    {
                        "type": "function_call",
                        "call_id": message.tool_call.call_id,
                        "name": message.tool_call.name,
                        "arguments": json.dumps(
                            dict(message.tool_call.arguments),
                            ensure_ascii=False,
                        ),
                    }
                )
            else:
                result.append({"role": message.role, "content": message.content})
        return result

    @staticmethod
    def _get(item: Any, key: str) -> Any:
        if isinstance(item, Mapping):
            return item.get(key)
        return getattr(item, key, None)


class AsyncOpenAIResponsesModel:
    """Async Responses API adapter used by the HTTP service."""

    def __init__(
        self,
        model: str,
        *,
        instructions: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - environment dependent.
                raise RuntimeError(
                    "Install the OpenAI extra: pip install -e '.[openai]'"
                ) from exc
            client = AsyncOpenAI()
        self._client = client
        self._model = model
        self._instructions = instructions

    @classmethod
    def from_env(cls, *, client: Any | None = None) -> "AsyncOpenAIResponsesModel":
        model = os.getenv("OPENAI_MODEL", "").strip()
        if not model:
            raise RuntimeError("OPENAI_MODEL is required")
        return cls(
            model,
            instructions=os.getenv("AGENT_INSTRUCTIONS") or None,
            client=client,
        )

    async def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> ModelResponse:
        request: dict[str, Any] = {
            "model": self._model,
            "input": OpenAIResponsesModel._to_input(messages),
            "tools": list(tools),
            "parallel_tool_calls": True,
        }
        if self._instructions:
            request["instructions"] = self._instructions
        response = await self._client.responses.create(**request)
        return OpenAIResponsesModel._parse_response(response)
