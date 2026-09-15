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

