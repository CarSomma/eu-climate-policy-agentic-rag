# Tool Framework TDD Plan

## Purpose

This document is the working TDD plan for finishing the tool framework
migration described in `docs/tool-framework-plan.md`.

Use this file as the day-to-day implementation checklist. Use
`tool-framework-plan.md` as the broader architecture and context document.

## Current State

The project is on the `tool-framework-tdd` branch. The migration has already
completed sixteen TDD slices.

The important completed milestones are:

- provider-neutral `core.tools` package exists
- `FunctionTool`, `BuiltinTool`, schema providers, `ToolRegistry`, and
  `ToolExecutor` exist
- sync and async execution are covered
- structured `ToolResult` and framework errors exist
- middleware exists and is used for fetch output-directory injection and RAG
  source collection
- async timeout, concurrency, cancellation, and retry behavior are covered
- per-tool `ToolExecutionConfig` exists
- OpenAI Responses function-calling loop is codified and extracted into
  `OpenAIResponsesToolLoop`
- first provider adapter exists:
  `core.tools.adapters.OpenAIResponsesToolAdapter`
- first provider schema compiler exists:
  `OpenAIResponsesSchemaCompiler`
- `AbstractAgent` now passes adapter-exported tools to
  `client.responses.create`
- legacy `core.tooling` still exists as a compatibility facade

Core compatibility expectations still in force:

- `OpenAIFunctionTool(...)`
- `ToolRegistry([tool])`
- `ToolRegistry(function_tools=[...], builtin_tools=[...])`
- `registry.schemas`
- `registry.run(...)`
- `registry.run_sync(...)`
- unknown local tools returning `{"error": "..."}`

## TDD Rules

For every slice:

1. Write or update tests first.
2. Run the smallest focused test command and confirm red.
3. Implement the smallest useful change.
4. Run the focused tests and confirm green.
5. Refactor only after green.
6. Run the relevant compatibility/integration tests.
7. Update `docs/tool-framework-plan.md` with a new completed slice.
8. Keep `docs/tool-framework-tdd.md` aligned if the next-slice ordering changes.

Useful default checks:

```bash
uv run pytest tests/unit
uv run pytest tests/integration/test_rag.py tests/integration/test_fetch_agent.py tests/integration/test_cleaning_agent.py
uv run ruff check src tests
```

Use narrower commands during red-green work.

## Next Slice 17: OpenAI Strict Schema Edge Cases

Goal: make strict schema failures explicit and local, instead of silently
stripping or emitting schemas that OpenAI may reject later.

Primary files:

- `src/eu_climate_policy_rag/core/tools/schema.py`
- `src/eu_climate_policy_rag/core/tools/adapters/openai_responses.py`
- `tests/unit/test_openai_responses_adapter.py`
- possibly a new `tests/unit/test_openai_schema_compiler.py`

Red tests:

- unsupported `dict[str, Any]` or open object raises `SchemaGenerationError`
- unsupported `allOf` raises `SchemaGenerationError`
- unsupported conditionals (`if` / `then` / `else`) raise `SchemaGenerationError`
- recursive `$ref` raises `SchemaGenerationError`
- invalid function tool name raises `SchemaGenerationError`

Green implementation:

- add validation to `OpenAIResponsesSchemaCompiler`
- keep normalization behavior for supported Pydantic schemas unchanged
- raise `SchemaGenerationError` with actionable messages for unsupported
  constructs

Acceptance:

- existing golden Responses schemas still pass
- existing RAG, fetch, and cleaning agent integrations still pass
- plan doc records the seventeenth TDD slice

## Slice 18: Schema Caching

Goal: avoid schema churn and repeated normalization on every model request.

Red tests:

- compiling the same `FunctionTool` twice reuses the cached compiled schema
- adapter `.tools` returns equal stable schemas across calls
- cache invalidation is not needed for immutable tool definitions

Green implementation:

- cache compiled Responses function schemas by `FunctionTool` instance or a
  stable schema fingerprint
- keep defensive copies if mutation risk appears in tests

Acceptance:

- no caller sees mutable shared schema state in a way that can corrupt later
  exports
- performance expectation in `tool-framework-plan.md` is now satisfied for
  OpenAI Responses export

## Slice 19: Native Tool Builders

Goal: start moving repo builders away from `core.tooling.OpenAIFunctionTool`.

Primary files:

- `src/eu_climate_policy_rag/qa/tools.py`
- `src/eu_climate_policy_rag/collection/fetching/fetch_tools.py`
- `src/eu_climate_policy_rag/collection/cleaning/cleaning_tools.py`

Red tests:

- builder modules return registries built from native `core.tools.FunctionTool`
- RAG `search_documents` behavior and source collection stay unchanged
- fetch save-directory middleware still applies
- cleaning agent direct tool calls still return the same values

Green implementation:

- replace `OpenAIFunctionTool(...)` builder usage with `FunctionTool(...)` plus
  `PydanticSchemaProvider(...)`
- keep the legacy facade available for external compatibility

Acceptance:

- no behavior changes in integration tests
- `rg "OpenAIFunctionTool" src/eu_climate_policy_rag` shows only compatibility
  module usage or intentionally deferred references

## Slice 20: Agent Boundary Uses Native Registry

Goal: reduce reliance on legacy `core.tooling.ToolRegistry` inside agents.

Red tests:

- `AbstractAgent` can accept the provider-neutral `core.tools.ToolRegistry`
- existing legacy `core.tooling.ToolRegistry` still works
- `AbstractAgent.tool_adapter` always receives a provider-neutral registry

Green implementation:

- accept either registry type at the agent boundary
- normalize once in `AbstractAgent.__init__`
- keep existing external behavior

Acceptance:

- RAG, fetch, and cleaning integration tests pass
- compatibility tests still pass

## Slice 21: Executor-Generated Responses Outputs

Goal: move local function-call handling toward `ToolExecutor` and `ToolResult`
instead of legacy raw registry return values.

Red tests:

- base `AbstractAgent` converts `ToolResult` through
  `OpenAIResponsesToolAdapter.to_function_call_output`
- unknown tools return model-visible structured errors in the base path
- domain overrides can still customize output shape where needed

Green implementation:

- add an executor-backed function-call dispatcher for the base agent
- preserve RAG text output and fetch JSON output through middleware or output
  mode choices

Acceptance:

- `function_call_output` shape is unchanged externally
- existing source collection and fetch logging still work

## Slice 22: Cleaning Middleware Review

Goal: decide whether cleaning needs middleware now, or explicitly mark it as no
immediate-op.

Possible middleware:

- cleaning result preview logging
- filesystem write authorization
- finalization metrics
- mutation guard around shared record state

Red tests if middleware is useful:

- middleware observes `save_cleaned_document`, `skip_document`, and `finalize`
- middleware does not change cleaning tool return values

Acceptance:

- either implemented middleware with tests, or plan doc says no current cleaning
  middleware is needed and why

## Slice 23: Deprecate `core.tooling`

Goal: make the compatibility layer visibly transitional.

Red tests:

- importing `OpenAIFunctionTool` and legacy `ToolRegistry` still works
- optional deprecation warning is emitted only where intended
- internal repo modules no longer import `core.tooling`

Green implementation:

- shrink `tooling.py` to a deprecation shim if all internal callers are native
- document replacement imports

Deletion gate:

```bash
rg "core.tooling|OpenAIFunctionTool|from .*tooling import" src tests
```

`tooling.py` can be deleted only when the search has no required internal or
test references left, except possibly a temporary deprecation test.

## Slice 24: Remove `core.tooling`

Goal: delete the legacy facade after migration.

Red tests:

- compatibility tests that require `core.tooling` are removed or rewritten
- all imports use `core.tools` and provider adapters directly

Green implementation:

- delete `src/eu_climate_policy_rag/core/tooling.py`
- remove compatibility tests that only preserve old import paths
- update docs and examples

Acceptance:

- full test suite passes
- README and docs no longer advertise legacy APIs

## Later Slices

These should wait until the core OpenAI Responses path is tidy:

- provider adapter for OpenAI Chat Completions, if needed
- provider adapter for Anthropic, Gemini, or MCP
- decorator registration API
- schema inspection helpers such as `registry.describe()`
- observability middleware and OpenTelemetry spans
- authorization/rate-limiting middleware
- tool call replay fixtures
- parallel function-call execution in `OpenAIResponsesToolLoop`

## Current Recommended Next Command

Start Slice 17 with focused tests:

```bash
uv run pytest tests/unit/test_openai_responses_adapter.py
```

Then add failing schema edge-case tests before changing implementation.
