# Design notes

## Purpose

This repository is a learning baseline that keeps the central agent loop visible.
Each later version adds one engineering pressure around that loop instead of
replacing it with a framework.

## Runtime data path

```text
caller
  -> Agent.run(user_input, optional history)
  -> ModelAdapter.respond(messages, tool schemas)
     -> final answer -> AgentRun
     -> ToolCall
        -> ToolRegistry JSON Schema validation
        -> handler execution
        -> ToolResult
        -> tool Message with call_id
        -> next model step
  -> max_steps -> AgentRun(answer=None)
```

The async path mirrors the same flow but bounds model and tool execution with
timeouts. Tools may retry using exponential backoff. The HTTP service uses this
async path. Trace events expose run, model, and tool boundaries without coupling
the loop to a logging vendor.

## Contract decisions

1. `ModelResponse` contains exactly one final answer or one tool call.
2. `ToolCall` and tool messages retain `call_id`, so a provider can pair a
   function result with the original request.
3. Tool arguments are validated before handlers run. Validation failures and
   handler failures become observations rather than process crashes.
4. Messages and result contracts are frozen data classes. The loop owns a mutable
   list internally and returns an immutable tuple.
5. Provider, trace, and memory components sit behind small protocols.
6. The API validates external input and does not return raw upstream exceptions.

## Context and memory

`Agent.run(..., history=...)` accepts prior messages but does not own persistence.
`InMemoryConversationStore` provides a process-local LRU session store with both a
session cap and a message cap. Orphaned function outputs are removed when a context
window cuts off the matching function call.

This is short-term conversation state, not semantic long-term memory. A production
implementation would replace the store protocol with durable storage and add
summarization or retrieval before the model request.

## OpenAI adapter

`OpenAIResponsesModel` translates local messages into Responses API input items.
Assistant tool calls become `function_call` items and tool observations become
`function_call_output` items. The adapter parses either the first function call or
the response's output text into the framework-neutral `ModelResponse` contract.

## Remaining limitations

- no persistent database or distributed session coordination;
- no per-session concurrency control, so overlapping writes can lose an update;
- no auth, quotas, rate limiting, or tenant isolation;
- no parallel custom tool execution;
- no streaming responses;
- no cancellation of already-running synchronous thread work after an async
  timeout;
- no live-model scoring, judge model, or statistical evaluation;
- no OpenTelemetry exporter or production deployment manifest.
