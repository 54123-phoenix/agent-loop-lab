from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_loop_lab import AsyncAgent, InMemoryTraceSink, ModelResponse
from agent_loop_lab.mock_model import AsyncScriptedModel
from agent_loop_lab.models import ToolCall
from agent_loop_lab.tools import ToolRegistry, ToolSpec, build_default_registry


SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


class SlowModel:
    async def respond(self, messages, tools):
        del messages, tools
        await asyncio.sleep(0.05)
        return ModelResponse.final("late")


class AsyncAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_direct_answer_and_emits_trace(self) -> None:
        tracer = InMemoryTraceSink()
        agent = AsyncAgent(
            AsyncScriptedModel([ModelResponse.final("done")]),
            build_default_registry(),
            tracer=tracer,
        )

        run = await agent.run("hello")

        self.assertEqual(run.answer, "done")
        self.assertEqual(
            [event.kind for event in tracer.events()],
            ["run_started", "model_requested", "run_finished"],
        )

    async def test_model_timeout_becomes_stop_reason(self) -> None:
        run = await AsyncAgent(
            SlowModel(),
            build_default_registry(),
            model_timeout_seconds=0.001,
        ).run("hello")

        self.assertEqual(run.stop_reason, "model_timeout")
        self.assertIn("timed out", run.error)

    async def test_async_tool_retries(self) -> None:
        attempts = 0

        async def flaky(arguments):
            nonlocal attempts
            del arguments
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary")
            return "recovered"

        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                "flaky",
                "Fail once",
                SCHEMA,
                flaky,
                max_attempts=2,
                retryable_exceptions=(RuntimeError,),
            )
        )
        result = await registry.execute_async(ToolCall("flaky", {}))

        self.assertTrue(result.ok)
        self.assertEqual(result.content, "recovered")
        self.assertEqual(result.attempts, 2)

    async def test_async_tool_timeout(self) -> None:
        async def slow(arguments):
            del arguments
            await asyncio.sleep(0.05)
            return "late"

        registry = ToolRegistry()
        registry.register(
            ToolSpec("slow", "Too slow", SCHEMA, slow, timeout_seconds=0.001)
        )

        result = await registry.execute_async(ToolCall("slow", {}))

        self.assertFalse(result.ok)
        self.assertIn("TimeoutError", result.content)
        self.assertEqual(result.error.code, "TOOL_TIMEOUT")
        self.assertTrue(result.error.retryable)


if __name__ == "__main__":
    unittest.main()
