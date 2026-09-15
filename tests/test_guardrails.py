from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_loop_lab import Agent, AsyncAgent, ModelResponse, RunBudget, ToolCall
from agent_loop_lab.mock_model import AsyncScriptedModel, ScriptedModel
from agent_loop_lab.tools import ToolRegistry, ToolSpec


SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


class GuardrailTests(unittest.TestCase):
    def test_side_effecting_tool_requires_approval(self) -> None:
        executions = 0

        def mutate(arguments):
            nonlocal executions
            del arguments
            executions += 1
            return "changed"

        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                "mutate",
                "Changes external state",
                SCHEMA,
                mutate,
                side_effect="destructive",
                idempotent=False,
            )
        )
        agent = Agent(
            ScriptedModel(
                [ModelResponse.call("mutate", {}), ModelResponse.final("done")]
            ),
            registry,
        )

        run = agent.run("change it")

        self.assertEqual(executions, 0)
        self.assertEqual(run.messages[2].tool_result.error.code, "APPROVAL_REQUIRED")

    def test_tool_call_budget_stops_a_batch_before_execution(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolSpec("noop", "No operation", SCHEMA, lambda arguments: "ok"))
        agent = Agent(
            ScriptedModel(
                [ModelResponse.call_many([ToolCall("noop", {}), ToolCall("noop", {})])]
            ),
            registry,
            budget=RunBudget(max_tool_calls=1),
        )

        run = agent.run("do too much")

        self.assertEqual(run.stop_reason, "max_tool_calls")
        self.assertEqual(len(run.messages), 1)


class ParallelToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_independent_tool_batch_runs_concurrently_in_order(self) -> None:
        active = 0
        peak_active = 0

        async def first(arguments):
            nonlocal active, peak_active
            del arguments
            active += 1
            peak_active = max(peak_active, active)
            await asyncio.sleep(0.05)
            active -= 1
            return "first"

        async def second(arguments):
            nonlocal active, peak_active
            del arguments
            active += 1
            peak_active = max(peak_active, active)
            await asyncio.sleep(0.05)
            active -= 1
            return "second"

        registry = ToolRegistry()
        registry.register(ToolSpec("first", "First", SCHEMA, first))
        registry.register(ToolSpec("second", "Second", SCHEMA, second))
        agent = AsyncAgent(
            AsyncScriptedModel(
                [
                    ModelResponse.call_many(
                        [ToolCall("first", {}), ToolCall("second", {})]
                    ),
                    ModelResponse.final("done"),
                ]
            ),
            registry,
        )

        run = await agent.run("run both")

        results = [
            message.tool_result.content
            for message in run.messages
            if message.role == "tool"
        ]
        self.assertEqual(results, ["first", "second"])
        self.assertEqual(peak_active, 2)
