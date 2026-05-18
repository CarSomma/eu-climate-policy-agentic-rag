# TDD Improvement Plan

## Purpose

This document captures the next useful additions and updates for the project
using a test-driven development workflow.

It follows the current state recorded in `docs/tool-framework-tdd.md` and
`docs/tool-framework-plan.md`: the provider-neutral tool framework migration is
complete through removal of the old `core.tooling` facade, and the next work
should focus on documentation alignment, framework ergonomics, test fixtures,
observability, and RAG quality.

## Current Status

- Completed: Slice 1, Documentation Alignment
- Completed: Slice 2, Tool Framework Developer Guide
- Completed: Slice 3, Registry Introspection
- Completed: Slice 4, Tool Call Replay Fixtures
- Completed: Slice 5, Parallel Async Function Calls
- Completed: Slice 6, Observability Middleware
- Next recommended slice: Slice 7, RAG Dataset Chunking
- Last verification: Slice 6 checks passed on 2026-05-18

## Progress Log

### 2026-05-12: Slice 1 Completed

Updated active documentation to match the completed tool-framework migration.

Changed files:

- `docs/tool-framework-tdd.md`
- `docs/package-structure.md`
- `docs/tests.md`
- `docs/tdd-improvement-plan.md`

Verification:

```bash
rg "core\.tooling|OpenAIFunctionTool|test_tooling" docs/package-structure.md docs/tests.md README.md
rg "Next Slice 17|Slice 18: Schema Caching|Slice 24: Remove" docs/tool-framework-tdd.md
uv run pytest tests/unit
```

### 2026-05-12: Slice 2 Completed

Added a contributor-facing guide for the provider-neutral tool framework.

Changed files:

- `docs/tools.md`
- `README.md`
- `docs/tdd-improvement-plan.md`
- `tests/integration/test_rag_dataset_ingestion.py`

Verification:

```bash
test -f docs/tools.md
rg "docs/tools.md|tools.md" README.md
uv run pytest tests/unit/test_tool_framework.py tests/unit/test_openai_responses_adapter.py
uv run ruff check src tests
uv run pytest tests/integration/test_rag_dataset_ingestion.py
```

### 2026-05-13: Slice 3 Completed

Added provider-neutral registry introspection for debugging, docs, and logs.

Changed files:

- `src/eu_climate_policy_rag/core/tools/registry.py`
- `tests/unit/test_tool_framework.py`
- `docs/tools.md`
- `docs/tdd-improvement-plan.md`

Verification:

```bash
uv run pytest tests/unit/test_tool_framework.py
uv run ruff check src tests
uv run pytest tests/unit
```

### 2026-05-13: Slice 4 Completed

Added reusable Responses-style replay fixtures for agent-loop regression tests.

Changed files:

- `tests/helpers/__init__.py`
- `tests/helpers/responses_replay.py`
- `tests/unit/test_agent_loop.py`
- `docs/tdd-improvement-plan.md`

Verification:

```bash
uv run pytest tests/unit/test_agent_loop.py
uv run ruff check tests/unit/test_agent_loop.py tests/helpers/responses_replay.py
uv run ruff check src tests
uv run pytest tests/unit
```

### 2026-05-13: Slice 5 Completed

Updated the async Responses tool loop to execute same-turn function calls
concurrently while preserving deterministic output order in message history.

Changed files:

- `src/eu_climate_policy_rag/core/agent_loop.py`
- `tests/unit/test_agent_loop.py`
- `docs/tdd-improvement-plan.md`

Verification:

```bash
uv run pytest tests/unit/test_agent_loop.py tests/unit/test_tool_executor.py
uv run ruff check src/eu_climate_policy_rag/core/agent_loop.py tests/unit/test_agent_loop.py
uv run pytest tests/integration/test_rag.py tests/integration/test_fetch_agent.py tests/integration/test_cleaning_agent.py
uv run ruff check src tests
uv run pytest tests/unit
```

### 2026-05-18: Slice 6 Completed

Added reusable observability metrics for local tool execution and model
Responses token usage. Cost estimates use configurable per-1M-token pricing
because provider pricing changes over time.

OpenAI docs checked:

- Responses include `usage.input_tokens`, `usage.output_tokens`, and
  `usage.total_tokens`.
- OpenAI publishes token pricing per 1M tokens and notes that tokens are billed
  at the selected model's input and output rates.

Changed files:

- `src/eu_climate_policy_rag/core/tools/middleware.py`
- `src/eu_climate_policy_rag/core/tools/executor.py`
- `src/eu_climate_policy_rag/core/tools/result.py`
- `src/eu_climate_policy_rag/core/tools/__init__.py`
- `src/eu_climate_policy_rag/core/agent_loop.py`
- `src/eu_climate_policy_rag/core/metrics.py`
- `tests/unit/test_tool_middleware.py`
- `tests/unit/test_response_metrics.py`
- `tests/unit/test_agent_loop.py`
- `docs/tools.md`
- `docs/tdd-improvement-plan.md`

Verification:

```bash
uv run pytest tests/unit/test_tool_middleware.py tests/unit/test_response_metrics.py tests/unit/test_agent_loop.py
uv run pytest tests/unit
uv run pytest tests/integration/test_rag.py tests/integration/test_fetch_agent.py tests/integration/test_cleaning_agent.py
uv run ruff check src tests
```

## TDD Workflow

For each slice:

1. Write or update the smallest focused test or consistency check first.
2. Run the focused command and confirm it fails for the expected reason.
3. Implement the smallest useful change.
4. Run the focused command again and confirm it passes.
5. Run the relevant compatibility checks.
6. Update docs if behavior, CLI usage, or public APIs changed.

Useful default checks:

```bash
uv run pytest tests/unit
uv run pytest tests/integration/test_rag.py tests/integration/test_fetch_agent.py tests/integration/test_cleaning_agent.py
uv run ruff check src tests
```

## Slice 1: Documentation Alignment

Status: completed on 2026-05-12.

Goal: remove stale references left behind by the completed tool-framework
migration.

Primary files:

- `docs/tool-framework-tdd.md`
- `docs/package-structure.md`
- `docs/tests.md`
- possibly `README.md`

Red checks:

```bash
rg "core\.tooling|OpenAIFunctionTool|test_tooling" docs/package-structure.md docs/tests.md README.md
rg "Next Slice 17|Slice 18: Schema Caching|Slice 24: Remove" docs/tool-framework-tdd.md
```

Expected failures:

- `docs/package-structure.md` still references `core/tooling.py`
- `docs/tests.md` still references `test_tooling.py`
- `docs/tool-framework-tdd.md` still reads like Slice 17 is the next active
  task, even though the plan doc records completion through Slice 24

Green implementation:

- update `docs/tool-framework-tdd.md` so it reflects the current completed
  migration state and points to this document for next work
- update `docs/package-structure.md` to describe `core/tools/`,
  `core/agent_loop.py`, adapters, executor, middleware, and schema providers
- update `docs/tests.md` to list current tool-framework tests:
  - `tests/unit/test_tool_framework.py`
  - `tests/unit/test_tool_executor.py`
  - `tests/unit/test_tool_middleware.py`
  - `tests/unit/test_openai_responses_adapter.py`

Acceptance:

```bash
rg "core\.tooling|OpenAIFunctionTool|test_tooling" docs/package-structure.md docs/tests.md README.md
uv run pytest tests/unit
```

The search should return no stale active docs references. Historical migration
notes may remain in `docs/tool-framework-plan.md`.

## Slice 2: Tool Framework Developer Guide

Status: completed on 2026-05-12.

Goal: add a concise guide for future contributors who need to create or execute
tools without reading the full migration plan.

Primary files:

- new `docs/tools.md`
- `README.md`

Red checks:

```bash
test -f docs/tools.md
rg "docs/tools.md|tools.md" README.md
```

Green implementation:

- document how to define a `FunctionTool`
- document how to use `PydanticSchemaProvider`
- document how to register `BuiltinTool.web_search`
- document how `ToolExecutor` returns `ToolResult`
- document how `OpenAIResponsesToolAdapter` exports OpenAI Responses schemas
- link `docs/tools.md` from the README documentation list

Acceptance:

```bash
uv run pytest tests/unit/test_tool_framework.py tests/unit/test_openai_responses_adapter.py
uv run ruff check src tests
```

## Slice 3: Registry Introspection

Status: completed on 2026-05-13.

Goal: add a lightweight way to inspect registered tools for debugging, docs,
and future observability.

Primary files:

- `src/eu_climate_policy_rag/core/tools/registry.py`
- `tests/unit/test_tool_framework.py`

Red tests:

- `ToolRegistry.describe()` returns function tool names and descriptions
- built-in tools are described separately from local function tools
- handler callables are not exposed in the description
- duplicate-name validation remains unchanged

Green implementation:

- add a simple `ToolRegistry.describe()` method returning serializable data
- include function names, descriptions, strict mode, and built-in tool types
- keep the output provider-neutral

Acceptance:

```bash
uv run pytest tests/unit/test_tool_framework.py
```

## Slice 4: Tool Call Replay Fixtures

Status: completed on 2026-05-13.

Goal: make agent-loop regressions easier to test without live API calls.

Primary files:

- `tests/unit/test_agent_loop.py`
- possibly `tests/fixtures/` or `tests/helpers/`

Red tests:

- replay a saved Responses-style sequence with multiple function calls
- replay unknown-tool output
- replay max-turn behavior
- confirm message history preserves response outputs and matching
  `function_call_output` items

Green implementation:

- add small fixture builders for Responses-style messages, function calls, and
  response objects
- keep fixtures local to tests unless they become broadly useful

Acceptance:

```bash
uv run pytest tests/unit/test_agent_loop.py
```

## Slice 5: Parallel Async Function Calls

Status: completed on 2026-05-13.

Goal: allow the async OpenAI Responses tool loop to execute multiple function
calls from a single response turn concurrently.

Primary files:

- `src/eu_climate_policy_rag/core/agent_loop.py`
- `tests/unit/test_agent_loop.py`
- `tests/unit/test_tool_executor.py`

Red tests:

- async loop executes multiple function calls from one model response
  concurrently
- one `function_call_output` is appended per tool call
- each output keeps the correct model-provided `call_id`
- errors are still returned in structured model-visible form
- sync loop behavior remains sequential

Green implementation:

- update `OpenAIResponsesToolLoop._append_response_output_async`
- gather same-turn async function calls while preserving deterministic output
  ordering in `message_history`
- keep sync behavior unchanged

Acceptance:

```bash
uv run pytest tests/unit/test_agent_loop.py tests/unit/test_tool_executor.py
uv run pytest tests/integration/test_rag.py tests/integration/test_fetch_agent.py tests/integration/test_cleaning_agent.py
```

## Slice 6: Observability Middleware

Status: completed on 2026-05-18.

Goal: add reusable tool execution observability without baking metrics into
domain agents.

Primary files:

- `src/eu_climate_policy_rag/core/tools/middleware.py`
- `src/eu_climate_policy_rag/core/tools/executor.py`
- `src/eu_climate_policy_rag/core/metrics.py`
- `tests/unit/test_tool_middleware.py`
- `tests/unit/test_response_metrics.py`

Red tests:

- middleware records tool name, attempt, success, and failure
- middleware can observe validation errors
- elapsed time is recorded without changing handler return values
- emitted metadata is attached to `ToolResult`
- model response token usage is captured from Responses `usage`
- total estimated cost is calculated from configurable model pricing

Green implementation:

- add a small callback-based metrics or event middleware
- add a response usage tracker with configurable per-model pricing
- keep it optional and provider-neutral
- avoid coupling to a specific telemetry backend

Acceptance:

```bash
uv run pytest tests/unit/test_tool_middleware.py tests/unit/test_tool_executor.py tests/unit/test_response_metrics.py
```

## Slice 7: RAG Dataset Chunking

Status: next.

Goal: improve retrieval quality by allowing cleaned documents to be split into
stable chunks before indexing.

Primary files:

- `src/eu_climate_policy_rag/collection/cleaning/rag_dataset_ingestion.py`
- possibly a new chunking module under `collection/cleaning/`
- `src/eu_climate_policy_rag/qa/tools.py`
- `tests/integration/test_rag_dataset_ingestion.py`
- `tests/integration/test_rag.py`

Red tests:

- long cleaned records can be chunked into multiple records
- each chunk preserves source title, URL, and stable chunk ID
- short records remain unchanged or produce one chunk
- RAG search can index chunked records

Green implementation:

- add deterministic chunking with stable IDs
- add an ingestion option such as `--chunk`
- keep the current whole-document behavior as the default unless tests and docs
  intentionally change it

Acceptance:

```bash
uv run pytest tests/integration/test_rag_dataset_ingestion.py tests/integration/test_rag.py
```

## Slice 8: Collection Pipeline Hardening

Goal: make the document collection workflow more predictable, inspectable, and
safe to rerun.

The `collection/` package owns discovery, URL normalization, duplicate
detection, quality checks, fetch orchestration, Markdown conversion, cleaning,
and ingestion. It should have its own TDD track because collection failures tend
to show up as missing or low-quality RAG context later.

Primary files:

- `src/eu_climate_policy_rag/collection/document_urls.py`
- `src/eu_climate_policy_rag/collection/document_quality.py`
- `src/eu_climate_policy_rag/collection/content_hashing.py`
- `src/eu_climate_policy_rag/collection/fetch_pipeline_steps.py`
- `src/eu_climate_policy_rag/collection/pipeline.py`
- `src/eu_climate_policy_rag/collection/cleaning/markdown_cleaning.py`
- `tests/unit/test_urls.py`
- `tests/unit/test_document_quality.py`
- `tests/unit/test_markdown_cleaning.py`
- `tests/integration/test_pipeline.py`
- `tests/integration/test_fetching_content.py`

Red tests:

- URL canonicalization handles more EU document variants:
  - EUR-Lex query URLs
  - EUR-Lex ELI URLs
  - language-specific URLs
  - download URLs with encoded filenames
- filename extraction sanitizes unsafe or awkward filenames while preserving
  readable document names
- duplicate detection treats whitespace-only and boilerplate-only differences
  as the same content
- document quality checks explain each rejection reason clearly enough for
  pipeline logs
- pipeline steps are idempotent when output files already exist
- fetch and cleaning steps can produce a compact run summary with counts for
  discovered, selected, fetched, skipped, duplicated, rejected, cleaned, and
  saved documents

Green implementation:

- expand `document_urls.py` around canonical keys, format detection, and safe
  filenames
- strengthen `content_hashing.py` normalization only where tests show real
  duplicate misses
- keep `DocumentQualityCheck` conservative but make reasons consistent and easy
  to assert in tests
- add a small collection run summary model or dataclass if existing return
  shapes are too loose
- thread summary data through `fetch_pipeline_steps.py` and `pipeline.py`
  without changing CLI defaults

Acceptance:

```bash
uv run pytest tests/unit/test_urls.py tests/unit/test_document_quality.py tests/unit/test_markdown_cleaning.py
uv run pytest tests/integration/test_pipeline.py tests/integration/test_fetching_content.py
```

## Slice 9: Collection Cleaning Quality Fixtures

Goal: protect the cleaning layer against regressions using realistic Markdown
fixtures.

Primary files:

- `src/eu_climate_policy_rag/collection/cleaning/markdown_cleaning.py`
- `tests/unit/test_markdown_cleaning.py`
- possibly `tests/fixtures/collection/`

Red tests:

- repeated PDF page headers and footers are removed
- short legal section headings are preserved
- tables and bullet lists survive cleaning
- EU boilerplate/navigation text is removed without deleting policy content
- cleaning is idempotent when run twice on the same Markdown

Green implementation:

- add focused fixture snippets that represent real fetched Markdown problems
- improve cleaning rules only where tests prove a regression or gap
- keep each rule small and explain non-obvious regexes with short comments

Acceptance:

```bash
uv run pytest tests/unit/test_markdown_cleaning.py
uv run pytest tests/integration/test_rag_dataset_ingestion.py
```

## Slice 10: CLI And Data Docs Refresh

Goal: make README and docs match the actual CLI behavior.

Primary files:

- `README.md`
- `docs/data.md`
- `docs/pipeline.md`
- `docs/tests.md`

Red checks:

```bash
uv run eu-climate-ask --help
uv run eu-climate-ingest --help
uv run eu-climate-pipeline --help
```

Green implementation:

- update `docs/data.md` with current `eu-climate-ask` options, including web
  search options
- update `docs/pipeline.md` if fetch or ingestion options have drifted
- update README examples if defaults or recommended commands changed

Acceptance:

```bash
uv run pytest tests/integration/test_pipeline.py tests/integration/test_rag.py
uv run ruff check src tests
```

## Recommended Order

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

Start with documentation alignment because it is low risk and prevents future
work from following stale migration instructions. After that, choose among
tool-framework polish, collection reliability, and RAG retrieval quality
depending on the project goal for the next development session.
