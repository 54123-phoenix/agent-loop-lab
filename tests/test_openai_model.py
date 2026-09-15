from __future__ import annotations

import sys
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_loop_lab.models import Message, ToolCall, ToolError, ToolResult
from agent_loop_lab.openai_model import AsyncOpenAIResponsesModel, OpenAIResponsesModel


class FakeResponses:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> object:
        self.requests.append(request)
        return self.response


class FakeClient:
    def __init__(self, response: object) -> None:
        self.responses = FakeResponses(response)


class AsyncFakeResponses(FakeResponses):
    async def create(self, **request: object) -> object:
        self.requests.append(request)
        return self.response


class AsyncFakeClient:
    def __init__(self, response: object) -> None:
        self.responses = AsyncFakeResponses(response)


class OpenAIResponsesModelTests(unittest.TestCase):
    def test_parses_function_call(self) -> None:
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="calculator",
                    call_id="call_123",
                    arguments='{"expression":"6 * 7"}',
                )
            ],
            output_text="",
        )
        client = FakeClient(response)
        model = OpenAIResponsesModel("test-model", client=client)

        result = model.respond([Message("user", "calculate")], [])

        self.assertEqual(result.tool_call.name, "calculator")
        self.assertEqual(result.tool_call.arguments, {"expression": "6 * 7"})
        self.assertEqual(result.tool_call.call_id, "call_123")
        self.assertFalse(client.responses.requests[0]["parallel_tool_calls"])

    def test_parses_final_text(self) -> None:
        response = SimpleNamespace(output=[], output_text="  done  ")
        model = OpenAIResponsesModel("test-model", client=FakeClient(response))

        result = model.respond([Message("user", "hello")], [])

        self.assertEqual(result.final_answer, "done")

    def test_serializes_tool_round_trip(self) -> None:
        response = SimpleNamespace(output=[], output_text="done")
        client = FakeClient(response)
        model = OpenAIResponsesModel("test-model", client=client)
        call = ToolCall("calculator", {"expression": "1 + 1"}, "call_1")
        messages = [
            Message("user", "calculate"),
            Message("assistant", "tool call", tool_call=call, call_id="call_1"),
            Message("tool", "ok=True content=2", "calculator", call_id="call_1"),
        ]

        model.respond(messages, [])

        request_input = client.responses.requests[0]["input"]
        self.assertEqual(request_input[1]["type"], "function_call")
        self.assertEqual(request_input[2]["type"], "function_call_output")

    def test_serializes_structured_tool_failure(self) -> None:
        response = SimpleNamespace(output=[], output_text="done")
        client = FakeClient(response)
        model = OpenAIResponsesModel("test-model", client=client)
        failure = ToolResult(
            name="calculator",
            ok=False,
            content="TimeoutError: slow",
            attempts=2,
            error=ToolError("TOOL_TIMEOUT", "TimeoutError: slow", retryable=True),
        )

        model.respond(
            [
                Message(
                    "tool",
                    failure.content,
                    "calculator",
                    call_id="call_1",
                    tool_result=failure,
                )
            ],
            [],
        )

        output = json.loads(client.responses.requests[0]["input"][0]["output"])
        self.assertEqual(output["error"]["code"], "TOOL_TIMEOUT")
        self.assertEqual(output["attempts"], 2)

    def test_rejects_invalid_arguments_json(self) -> None:
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="calculator",
                    call_id="call_123",
                    arguments="not-json",
                )
            ],
            output_text="",
        )
        model = OpenAIResponsesModel("test-model", client=FakeClient(response))

        with self.assertRaises(RuntimeError):
            model.respond([Message("user", "calculate")], [])

    def test_rejects_function_call_without_call_id(self) -> None:
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="calculator",
                    call_id=None,
                    arguments='{"expression":"1 + 1"}',
                )
            ],
            output_text="",
        )
        model = OpenAIResponsesModel("test-model", client=FakeClient(response))

        with self.assertRaises(RuntimeError):
            model.respond([Message("user", "calculate")], [])


class AsyncOpenAIResponsesModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_parses_final_text(self) -> None:
        response = SimpleNamespace(output=[], output_text="done")
        model = AsyncOpenAIResponsesModel(
            "test-model",
            client=AsyncFakeClient(response),
        )

        result = await model.respond([Message("user", "hello")], [])

        self.assertEqual(result.final_answer, "done")


if __name__ == "__main__":
    unittest.main()
