# From V0.1 to V0.5

Read this file beside the code. Each version answers one question that the previous
version cannot answer.

## V0.1: can the loop be seen?

Core files: `models.py`, `tools.py`, `agent.py`, and `mock_model.py`.

```text
messages = [user message]
repeat at most max_steps:
    response = model.respond(messages, tool schemas)
    if response is an answer: return it
    result = tools.execute(response.tool_call)
    append result to messages
return max_steps
```

Protection: a hard stop, explicit response states, and tool failures represented as
observations. Limitation: the scripted model ignores context and tool schemas.

## V0.2: can untrusted boundaries be checked?

Added `openai_model.py`. `ToolSpec` now contains a full JSON Schema. `ToolRegistry`
validates every argument object before calling a handler. `ToolCall` and `Message`
retain provider call IDs.

Effect:

```text
before: {"expression": 42} reaches calculate() after implicit str conversion
after:  rejected because expression must be a string
```

The OpenAI adapter is isolated from the loop. Replacing providers should require a
new adapter, not changes to `Agent` or tool handlers.

## V0.3: can a change be measured?

Added `evaluation.py`, `evals/basic.jsonl`, and `examples/run_evals.py`.

Each versioned case can assert:

- exact answer when deterministic;
- stop reason;
- ordered tools used.

The runner reports pass count, pass rate, and average model steps. This catches
contract regressions, but a tiny scripted set is not evidence of model quality.

## V0.4: can slow and transient work be contained?

Added `async_agent.py` and `tracing.py`. `ToolSpec` adds timeout, retry count, and
backoff settings.

```text
model call -> timeout -> model_timeout AgentRun
tool call  -> fail -> retry -> success/failure observation
run        -> trace events for start, model request, tool start/end, finish
```

Synchronous tools can retry. Enforceable timeouts live on the async execution path;
Python cannot safely kill arbitrary synchronous work already running in a thread.

## V0.5: can more than one user turn be retained safely?

Added `memory.py`, `api.py`, input models, and deployment files.

```text
HTTP request
  -> boundary validation
  -> load bounded session history
  -> Agent.run(message, history=history)
  -> save bounded result history
  -> stable HTTP response
```

Two limits prevent unbounded in-process growth: maximum sessions and maximum
messages per session. `reset=true` starts a fresh session history. This is a
teaching implementation: restart the process and the memory disappears.

## What changed conceptually?

The central loop barely changed. Most new code protects a boundary:

```text
model boundary        -> adapter and response parsing
tool boundary         -> JSON Schema
behavior boundary     -> evaluations
time boundary         -> timeout and retry
visibility boundary   -> tracing
conversation boundary -> bounded memory
network boundary      -> FastAPI validation and stable errors
deployment boundary   -> Docker image
```

That is the main lesson of the version progression: engineering complexity usually
accumulates around a small piece of business logic, not inside it.

## V0.6: can full history and model context be separated?

Added `context.py` and `SQLiteConversationStore`.

- SQLite keeps the complete append-only message history across process restarts.
- `ContextManager` selects only the newest complete message/tool units that fit an
  explicit input budget.
- Dropped history can be represented by a bounded local summary without deleting
  the original events.
- Trace events report selected message count, dropped message count, and estimated
  input tokens.

The default token estimator is deliberately approximate and dependency-free. A
provider tokenizer can replace it through the `TokenEstimator` protocol.

## V0.7: can concurrent requests avoid losing or duplicating messages?

Added `coordination.py`, versioned conversation snapshots, and API `request_id`.

- Each session has an optimistic version; stale saves fail with a conflict instead
  of silently replacing newer history.
- `SessionCoordinator` serializes requests for one session inside a process while
  allowing different sessions to run concurrently.
- Repeating the same `request_id` and payload returns the recorded response without
  running the model or appending messages again.
- Reusing a `request_id` with different input returns an idempotency conflict.
- SQLite request journals preserve idempotency records across restarts.

## V0.8: can tool failures be handled as data?

Tool results now retain stable error codes, retryability, attempt count, duration,
and structured data. The provider receives that structure as JSON instead of
having to infer failure state from a sentence.

- Only explicitly classified transient exceptions are retried.
- Non-idempotent tools cannot enable automatic retries.
- SQLite history preserves the complete structured result.
- Tool side-effect metadata prepares the next layer's approval boundary.

## V0.9: can one run do more work without losing control?

One model step may now request multiple independent tools. The async agent runs
approved calls concurrently, then returns observations in the model's original
order.

- `RunBudget` bounds model steps, total tool calls, and wall time.
- Side-effecting tools are denied by default unless an approval policy allows them.
- Oversized tool output is truncated before it enters history or model context.
- A batch that would exceed its tool-call budget is rejected before any handler runs.
