# Tool Framework TDD Status

## Purpose

This document records the current status of the OpenAI Responses tool framework
TDD migration described in `docs/tool-framework-plan.md`.

The migration itself is complete. Use `docs/tool-framework-plan.md` for the
historical architecture record, and use `docs/tdd-improvement-plan.md` for the
active next slices.

## Current State

The provider-neutral tool framework is now the active implementation.

Completed milestones:

- provider-neutral `core.tools` package exists
- `FunctionTool`, `BuiltinTool`, schema providers, `ToolRegistry`, and
  `ToolExecutor` exist
- sync and async local tool execution are covered
- structured `ToolResult` and framework errors exist
- middleware exists and is used for fetch output-directory injection, RAG source
  collection, and cleaning metrics
- async timeout, concurrency, cancellation, and retry behavior are covered
- per-tool `ToolExecutionConfig` exists
- OpenAI Responses function-calling loop is codified and extracted into
  `OpenAIResponsesToolLoop`
- OpenAI Responses provider adapter exists:
  `core.tools.adapters.OpenAIResponsesToolAdapter`
- OpenAI Responses schema compiler exists:
  `OpenAIResponsesSchemaCompiler`
- `AbstractAgent` passes adapter-exported tools to `client.responses.create`
- `AbstractAgent` accepts native `core.tools.ToolRegistry`
- domain tool builders use native `FunctionTool` and `PydanticSchemaProvider`
- the old compatibility facade has been removed

Current core expectations:

- `FunctionTool(...)`
- `ToolRegistry(function_tools=[...], builtin_tools=[...])`
- `OpenAIResponsesToolAdapter(registry).tools`
- `ToolExecutor(registry).run(...)`
- `ToolExecutor(registry).run_sync(...)`
- base-agent unknown tools return structured model-visible errors
- domain direct-dispatch methods preserve their existing public return shapes

## TDD Rules

For every future slice:

1. Write or update tests first.
2. Run the smallest focused test command and confirm red.
3. Implement the smallest useful change.
4. Run the focused tests and confirm green.
5. Refactor only after green.
6. Run the relevant compatibility or integration tests.
7. Update the relevant docs after behavior, CLI usage, or public APIs change.
8. Keep `docs/tdd-improvement-plan.md` aligned if slice ordering changes.

Useful default checks:

```bash
uv run pytest tests/unit
uv run pytest tests/integration/test_rag.py tests/integration/test_fetch_agent.py tests/integration/test_cleaning_agent.py
uv run ruff check src tests
```

Use narrower commands during red-green work.

## Active Next Work

The current recommended next slices live in `docs/tdd-improvement-plan.md`.

At the time this status file was updated, the active order starts with:

1. Documentation Alignment
2. Tool Framework Developer Guide
3. Registry Introspection
4. Tool Call Replay Fixtures
5. Parallel Async Function Calls
6. Observability Middleware
7. RAG Dataset Chunking
8. Collection Pipeline Hardening
9. Collection Cleaning Quality Fixtures
10. CLI And Data Docs Refresh
