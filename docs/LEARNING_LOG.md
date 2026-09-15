# Learning log

## 2026-09-14 - Agent loop V0.1 understanding and layered expansion

- Identified that V0.1 has only per-run append-only message history, no cross-run
  memory, no context trimming, and no real model behavior.
- Distinguished user conversation turns from internal agent/model steps.
- Reduced the design to user message -> model decision -> optional tool ->
  observation -> next step or final answer.
- Recognized that most apparent complexity comes from engineering contracts and
  safeguards around a small loop.
- Expanded the repository through V0.2-V0.5 to compare strict schemas, a provider
  adapter, evaluation, reliability, tracing, bounded memory, and an API.

Next learning target: independently modify one layer and explain the failure it is
designed to prevent, rather than memorizing every Python type annotation.

## 2026-09-15 - Agent loop V0.6-V0.10 hardening

- Separated complete conversation history from the token-limited model context.
- Added durable SQLite history, optimistic session versions, per-session
  coordination, and idempotent HTTP requests.
- Replaced string-only tool failures with stable error data and side-effect-safe
  retry rules.
- Added parallel tool batches while bounding steps, tool count, wall time, output,
  and approval of side effects.
- Separated long-term facts/preferences/goals from chat history; added scoped
  retrieval, provenance, confidence, expiry, and deletion.

Next learning target: choose one failure scenario (duplicate request, stale write,
unsafe retry, context overflow, or false memory), reproduce it with a test, and
explain which single protection stops it.
