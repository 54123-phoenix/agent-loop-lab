# Design notes

## Purpose

This repository is a learning baseline that keeps the central agent loop visible.
Each later version adds one engineering pressure around that loop instead of
replacing it with a framework.

## Runtime data path

```text
caller
  -> retrieve relevant structured memories
  -> Agent.run(user_input, optional history, optional memories)
  -> ContextManager applies history/summary/memory token budgets
  -> ModelAdapter.respond(messages, tool schemas)
     -> final answer -> AgentRun
     -> one or more ToolCalls
        -> run/tool-count/time budget checks
        -> side-effect approval policy
        -> ToolRegistry JSON Schema validation
        -> parallel async handler execution
        -> structured ToolResult
        -> tool Message with call_id
        -> next model step
  -> max_steps -> AgentRun(answer=None)
```

The async path mirrors the same flow but bounds model and tool execution with
timeouts. Tools may retry using exponential backoff. The HTTP service uses this
async path. Trace events expose run, model, and tool boundaries without coupling
the loop to a logging vendor.

## Contract decisions

1. `ModelResponse` contains exactly one final answer, one tool call, or one call batch.
2. `ToolCall` and tool messages retain `call_id`, so a provider can pair a
   function result with the original request.
3. Tool arguments are validated before handlers run. Validation failures and
   handler failures become observations rather than process crashes.
4. Messages and result contracts are frozen data classes. The loop owns a mutable
   list internally and returns an immutable tuple.
5. Provider, trace, and memory components sit behind small protocols.
6. The API validates external input and does not return raw upstream exceptions.

## Context and memory

Conversation events and long-term memories have different jobs. The event store is
append-only evidence of what happened. `ContextManager` selects a bounded recent
window and an optional summary. The long-term store keeps small structured records,
retrieves only relevant live records for one owner, and supports explicit deletion.
SQLite implementations survive restart; in-memory implementations keep tests and
local demos simple.

## OpenAI adapter

`OpenAIResponsesModel` translates local messages into Responses API input items.
Assistant tool calls become `function_call` items and tool observations become
`function_call_output` items. The adapter parses either the first function call or
the response's output text into the framework-neutral `ModelResponse` contract.
Multiple function calls are preserved as one batch.

## Remaining limitations

- no auth, quotas, rate limiting, or tenant isolation;
- no distributed session lock across multiple server processes;
- no vector retrieval or automatic memory extraction/consolidation;
- no streaming responses;
- no cancellation of already-running synchronous thread work after an async
  timeout;
- no live-model scoring, judge model, or statistical evaluation;
- no OpenTelemetry exporter or production deployment manifest.
