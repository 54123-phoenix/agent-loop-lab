# Design notes

## Why this repository exists

This project is a learning baseline, not a production agent framework. Its goal is to make the loop visible and testable before adding an external LLM SDK or framework.

## Current flow

```text
User message
  → model adapter
  → final answer ───────────────→ stop
  → tool call
      → registry validation
      → tool execution
      → observation appended
      → next model step
  → max_steps reached ──────────→ stop without an answer
```

## Decisions

1. **No LLM API in V0.1.** A deterministic scripted model makes loop behavior reproducible and keeps secrets out of the first commit.
2. **Explicit response contract.** A model response must contain exactly one final answer or one tool call.
3. **Tool failures are observations.** Unknown tools, missing arguments, and handler errors are returned to the loop instead of crashing it.
4. **A hard step limit exists.** Repeated tool calls stop with `max_steps`; the program does not pretend it produced a final answer.
5. **The calculator parses an AST.** It does not use `eval`, and it rejects names, calls, and unsupported nodes.

## Known limitations

- No real LLM adapter or structured provider response parser.
- Tool schemas are deliberately minimal and do not yet use JSON Schema.
- No async tools, retries, timeouts, tracing, persistence, approval gates, or token budget.
- Tests validate deterministic behavior, not answer correctness from a live model.

## Next version

V0.2 should add one real provider adapter behind the same protocol, JSON Schema tool contracts, timeouts, and a versioned evaluation dataset.
