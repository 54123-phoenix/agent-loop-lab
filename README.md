# agent-loop-lab

A minimal, framework-free Python agent loop built to make **tool calling, observations, stop conditions, and failures** explicit.

> Status: V0.1 learning scaffold. It uses a deterministic scripted model, not a live LLM. Local verification: **11/11 tests passed on Python 3.11.9**.

## Why

Agent frameworks hide useful details. This repository starts with the smallest inspectable loop, then adds engineering features only when each one can be explained and tested.

## Current scope

- explicit `final_answer` versus `tool_call` response contract;
- tool registry with unknown-tool and missing-argument handling;
- calculator and word-count example tools;
- tool observations appended to message history;
- hard `max_steps` stop condition;
- deterministic model adapter for reproducible tests;
- standard-library unit tests and no runtime dependencies.

Not included yet: live LLM APIs, RAG, memory persistence, async execution, approvals, tracing, FastAPI, or deployment.

## Architecture

```text
User input
   ↓
Agent.run
   ↓
ModelAdapter.respond ──→ final answer ──→ stop
   │
   └─→ tool call → ToolRegistry → ToolResult/Observation
                                └───────────────→ next step
```

See [docs/DESIGN.md](docs/DESIGN.md) for decisions and limitations.

## Run the demo

```bash
python examples/demo.py
```

Expected final lines:

```text
[assistant] 6 × 7 = 42
stop_reason=final_answer, steps=2
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

Verified on 2026-09-08 with Python 3.11.9: **11 tests passed**. The demo completed in two steps and stopped with `final_answer`.

## Learning questions this V0.1 answers

1. Why is an agent a loop?
2. How are tools described to the model?
3. What can the model return?
4. How does code distinguish an answer from a tool call?
5. Where does the observation go?
6. Why and when does the loop stop?

## Roadmap

- [x] V0.1: deterministic loop, two tools, errors, stop condition, tests
- [ ] V0.2: real LLM adapter + JSON Schema tool contracts
- [ ] V0.3: evaluation dataset and metrics
- [ ] V0.4: async tools, retries, timeouts, tracing
- [ ] V0.5: FastAPI interface and deployment

## Honesty statement

This is a learning project. Features are marked complete only after they have runnable code and verification. Reference repositories may inform later iterations, but their implementation and results are not presented as this project's work.
