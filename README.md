# agent-loop-lab

A small Python agent loop that grows in visible layers from a deterministic V0.1
learning scaffold to a guarded V0.10 service with durable, retrievable memory.

The original loop is still the center of the project:

```text
user message
  -> model decision
  -> final answer -----------------------> stop
  -> tool call -> validation -> execution
              -> observation ------------> next model step
  -> max_steps --------------------------> stop without an answer
```

## Version layers

| Version | Added layer | Main protection |
| --- | --- | --- |
| V0.1 | deterministic loop and two tools | explicit stop conditions |
| V0.2 | OpenAI Responses adapter and JSON Schema | provider boundary and strict tool input |
| V0.3 | JSONL evaluation cases and metrics | behavior changes become measurable |
| V0.4 | async loop, timeouts, retries, tracing | slow and transient failures are bounded |
| V0.5 | FastAPI and bounded session memory | validated HTTP input and controlled context growth |
| V0.6 | SQLite event history and ContextManager | durable history and token-budgeted model context |
| V0.7 | session coordination and request journal | concurrency conflicts and retry duplication |
| V0.8 | structured tool results and retry policy | stable errors and side-effect-safe retries |
| V0.9 | parallel tool batches, approvals, and run budgets | bounded cost and controlled side effects |
| V0.10 | structured long-term memory and retrieval | relevant recall, expiry, provenance, and deletion |

See [docs/VERSIONS.md](docs/VERSIONS.md) for the code path and trade-offs of each
layer. This is still a learning project rather than a production framework.

## Install

Core loop only:

```bash
python -m pip install -e .
```

Full project, API, and test dependencies:

```bash
python -m pip install -e ".[api,dev]"
```

## Run deterministic examples

```bash
python examples/demo.py
python examples/run_evals.py
python -m unittest discover -s tests -v
```

These commands do not make network requests and do not require an API key.

## Run with the OpenAI Responses API

Copy `.env.example` values into your environment and set real values without
committing them:

```bash
set OPENAI_API_KEY=your-key
set OPENAI_MODEL=a-model-available-to-your-project
python examples/real_openai_demo.py
```

The adapter sends structured function definitions, preserves function call IDs,
and feeds function outputs back into the next model request. Independent function
calls from one model response run concurrently on the async path while preserving
their original order.

## Run the HTTP API

```bash
uvicorn agent_loop_lab.api:app --host 127.0.0.1 --port 8000
```

Example request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat `
  -ContentType "application/json" `
  -Body '{"session_id":"demo","message":"Calculate 6 * 7"}'
```

Endpoints:

- `GET /health`
- `POST /v1/chat`
- `GET /v1/sessions/{session_id}`
- `DELETE /v1/sessions/{session_id}`
- `POST /v1/sessions/{session_id}/memories`
- `GET /v1/sessions/{session_id}/memories`
- `DELETE /v1/sessions/{session_id}/memories/{memory_id}`

Set `AGENT_DB_PATH` to use one SQLite database for append-only conversation events,
request idempotency records, and long-term memories. Without it, both stores are
process-local. Long-term memories are added explicitly, retrieved by owner and
query, and injected into a separate bounded context section.

## Docker

```bash
docker build -t agent-loop-lab .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY \
  -e OPENAI_MODEL \
  agent-loop-lab
```

## Current boundaries

- Retrieval is deterministic keyword matching, not embedding/vector search.
- Memories are explicit records; automatic extraction and consolidation are not
  enabled because unverified model guesses should not silently become durable facts.
- Session IDs scope data but are not authentication or tenant isolation.
- Same-process session requests are serialized; multi-process coordination would
  require a shared lock or queue.
- There is no rate limiting, distributed tracing, streaming, or production secret
  manager. Do not expose this learning service directly to the public internet.
- The deterministic evaluation set checks contracts and regressions; it does not
  establish live-model quality.
