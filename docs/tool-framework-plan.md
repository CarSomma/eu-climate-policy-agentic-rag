# OpenAI Responses Tool Framework Plan

## Purpose

This document is the engineering plan for replacing the current small
`OpenAIFunctionTool` and `ToolRegistry` helpers with a production-grade tool
framework for the OpenAI Responses API.

The framework should continue to serve the current repository use cases:

- RAG question answering with `search_documents`
- Optional OpenAI built-in `web_search`
- Document fetching tools, including async browser-driven handlers
- Cleaning curation tools, including sync filesystem-backed handlers

The first concrete provider target is OpenAI Responses API, because that is
what the repository uses today. The core execution framework should not be
OpenAI-only, though. OpenAI-specific schema compilation and tool export should
live behind provider adapters so the same local tools can later be exported to
other LLM providers where practical.

It should also be designed so future tool definitions are not locked to
Pydantic. Pydantic v2 should remain the first-class local validation path, but
the core framework should work with any schema provider that can expose:

- an OpenAI-compatible JSON Schema
- a runtime argument validator
- an optional typed Python input object

## Current Repo Surface

Current shared implementation:

- `src/eu_climate_policy_rag/core/tooling.py`
  - `OpenAIFunctionTool`
  - `ToolRegistry`
  - Pydantic validation through `input_model.model_validate`
  - schema generation through `input_model.model_json_schema`
  - sync and async dispatch
  - built-in tool passthrough via raw dictionaries

Current active consumers:

- `src/eu_climate_policy_rag/core/agent.py`
  - sends `self.tools.schemas` to `client.responses.create`
  - handles `function_call` messages
  - appends `function_call_output`
- `src/eu_climate_policy_rag/qa/rag.py`
  - registers `search_documents`
  - optionally registers built-in `web_search`
  - overrides tool execution to collect RAG sources
  - emits plain text context for search results
- `src/eu_climate_policy_rag/qa/tools.py`
  - wraps `SearchDocumentsTool` as `OpenAIFunctionTool`
- `src/eu_climate_policy_rag/collection/fetching/fetch_tools.py`
  - builds async and sync fetch tools
- `src/eu_climate_policy_rag/collection/fetching/fetch_agent.py`
  - async dispatch
  - injects `directory` into `save_content_to_file`
  - serializes all tool outputs as JSON
- `src/eu_climate_policy_rag/collection/cleaning/cleaning_tools.py`
  - builds sync cleaning tools
- `src/eu_climate_policy_rag/collection/cleaning/cleaning_agent.py`
  - sync dispatch

Existing tests depend on:

- `OpenAIFunctionTool` import path
- `ToolRegistry` import path
- `ToolRegistry([...])` legacy constructor
- `ToolRegistry(function_tools=[...], builtin_tools=[...])`
- `registry.schemas`
- unknown tools returning `{"error": "..."}`
- built-in tools appearing in schemas but not in local dispatch

## Implementation Status

Current branch: `tool-framework-tdd`

Completed in the first TDD slice:

- created `docs/tool-framework-plan.md`
- added provider-neutral `core.tools` package
- added `FunctionTool`
- added `BuiltinTool`
- added `SchemaProvider`
- added `PydanticSchemaProvider`
- added `RawJsonSchemaProvider`
- added initial OpenAI strict schema normalization
- added provider-neutral `ToolRegistry`
- added `.openai_tools` and `.schemas` compatibility export
- added tests in `tests/unit/test_tool_framework.py`
- verified existing `tests/unit/test_tooling.py` still passes

Completed in the second TDD slice:

- added `ToolExecutor`
- added `ToolContext`
- added `ToolResult`
- added structured framework exceptions
- added sync execution pipeline for local function tools
- added structured success serialization
- added structured unknown-tool error serialization
- added structured validation error serialization
- added Responses `function_call_output` conversion on `ToolResult`
- added tests in `tests/unit/test_tool_executor.py`

Completed in the third TDD slice:

- added async `ToolExecutor.run`
- added support for async function handlers
- added async execution support for existing sync handlers through
  `asyncio.to_thread`
- added async structured validation error handling
- added async unknown-tool raise/return behavior
- added sync-mode guard for async handlers
- expanded `tests/unit/test_tool_executor.py`

Completed in the fourth TDD slice:

- migrated `core.tooling.OpenAIFunctionTool` to delegate to `FunctionTool`
- migrated `core.tooling.ToolRegistry` to delegate to the new registry and
  executor
- preserved legacy raw handler return values from `run` and `run_sync`
- preserved legacy unknown-tool return shape
- added `openai_tools` alias on the compatibility registry
- upgraded compatibility schemas to strict Responses-style export
- added Pydantic result-model serialization in `ToolResult`
- verified RAG, web-search, cleaning-agent, and fetch-agent integration paths

Completed in the fifth TDD slice:

- added `ToolMiddleware`
- added sync middleware hooks around validation and handler calls
- added async execution coverage for the same middleware lifecycle
- allowed middleware to mutate arguments before validation
- allowed middleware to mutate call arguments before handler invocation
- allowed middleware to mutate/observe handler results after execution
- propagated context metadata into `ToolResult`
- added tests in `tests/unit/test_tool_middleware.py`

Completed in the sixth TDD slice:

- added middleware support to the legacy `core.tooling.ToolRegistry` facade
- added `SaveContentDirectoryMiddleware` for fetch tools
- moved fetch `save_content_to_file` directory override from agent dispatch into
  middleware
- simplified `DocumentFetchAgent._run_tool_by_name`
- added integration coverage proving direct registry dispatch applies the fetch
  output directory middleware

Completed in the seventh TDD slice:

- added `SearchDocumentsResultMiddleware` for RAG search tools
- moved RAG source collection from `ClimatePolicyAgent._execute_tool_call` into
  middleware
- converted `SearchDocumentsResultModel` into model-facing context text through
  middleware
- simplified `ClimatePolicyAgent._execute_tool_call`
- changed per-answer source reset to clear the existing source list so
  middleware retains the correct sink
- added integration coverage proving registry dispatch collects RAG sources

Completed in the eighth TDD slice:

- added executor-level async timeout support through `timeout_seconds`
- wrapped async handler execution in `asyncio.timeout`
- returned structured `ToolExecutionError` timeout results in return-error mode
- raised `ToolExecutionError` timeout exceptions in raise-error mode
- added timeout pass-through on the legacy `core.tooling.ToolRegistry` facade
- expanded `tests/unit/test_tool_executor.py`

Completed in the ninth TDD slice:

- added executor-level async concurrency limiting through `max_concurrency`
- wrapped async handler execution in an `asyncio.Semaphore`
- validated `max_concurrency` configuration
- added `max_concurrency` pass-through on the legacy
  `core.tooling.ToolRegistry` facade
- added unit coverage proving concurrent async calls serialize when
  `max_concurrency=1`

Completed in the tenth TDD slice:

- added cancellation propagation coverage for async tool execution
- verified `asyncio.CancelledError` is not converted into a normal tool result
- verified concurrency slots are released after cancellation
- added per-call async timeout override on `ToolExecutor.run`
- expanded `tests/unit/test_tool_executor.py`

Completed in the eleventh TDD slice:

- added executor-level async retry support through `max_retries`
- retried handler execution failures after validation succeeds
- kept validation failures outside the retry loop
- returned structured `ToolExecutionError` after retry exhaustion
- added `max_retries` pass-through on the legacy `core.tooling.ToolRegistry`
  facade
- expanded `tests/unit/test_tool_executor.py`

Completed in the twelfth TDD slice:

- added `ToolExecutionConfig` for per-tool execution policy
- added per-tool async retry configuration
- added per-tool async timeout configuration
- added per-tool async concurrency limiting
- kept per-call async timeout overrides highest precedence
- kept executor-level timeout, retry, and concurrency defaults as fallbacks
- added `ToolExecutionConfig` support to legacy `OpenAIFunctionTool`
- expanded `tests/unit/test_tool_executor.py` and `tests/unit/test_tooling.py`

Completed in the thirteenth TDD slice:

- codified the core OpenAI Responses function-calling loop from the official
  function-calling guide:
  - preserve `response.output` in the running input/message history
  - execute every `function_call` item returned by the model
  - append one `function_call_output` item per tool call using the matching
    `call_id`
  - continue the loop with the augmented input until the model returns a
    message or the turn limit is reached
- added focused unit coverage in `tests/unit/test_agent_loop.py`
- centralized sync and async response-output handling in `core.agent`
- preserved existing domain-specific tool dispatch overrides for RAG, fetch,
  and cleaning agents

Completed in the fourteenth TDD slice:

- extracted the Responses function-calling loop into `core.agent_loop`
- added `OpenAIResponsesToolLoop` as a reusable sync/async loop object
- kept the loop callback-based so agents still own response creation,
  tool-call execution, message hooks, and built-in-tool logging
- migrated `AbstractAgent._run_loop` and `_run_loop_async` to delegate to the
  reusable loop object
- preserved the existing `(final_answer, message_history)` return shape
- expanded `tests/unit/test_agent_loop.py`
- verified existing RAG, fetch, and cleaning agent integrations still pass

Completed in the fifteenth TDD slice:

- added `core.tools.adapters` package
- added `OpenAIResponsesToolAdapter` as the first provider-specific tool adapter
- moved model-visible Responses tool export behind
  `OpenAIResponsesToolAdapter.tools`
- added adapter compatibility aliases `.openai_tools` and `.schemas`
- added adapter conversion from `ToolResult` to Responses
  `function_call_output`
- added a public `ToolRegistry.builtins` accessor for normalized built-in
  provider tools
- added a `base_registry` accessor on the legacy `core.tooling.ToolRegistry`
  facade
- updated `AbstractAgent` to pass adapter-exported tools to
  `client.responses.create`
- added coverage in `tests/unit/test_openai_responses_adapter.py` and expanded
  `tests/unit/test_agent_loop.py`

Not done yet:

- cleaning middleware opportunities, if any
- broader OpenAI strict schema edge-case handling
- move strict schema compilation fully into the OpenAI Responses adapter/compiler
- provider adapter modules beyond OpenAI Responses

## Goals

### Design Goals

- Provide a reliable local tool execution layer for OpenAI Responses API
  function calls.
- Generate strict OpenAI-compatible tool schemas before request time.
- Separate local executable function tools from OpenAI built-in tools.
- Support Pydantic v2 models without making Pydantic the only schema source.
- Keep local tool execution provider-neutral.
- Isolate provider-specific tool export behind adapters.
- Support sync and async execution with predictable behavior.
- Preserve backward compatibility during migration.
- Provide typed registration, typed dispatch, and structured results.
- Add structured errors that can either raise or be serialized back to the
  model.
- Add middleware and lifecycle hooks for logging, authorization, argument
  mutation, metrics, and tracing.
- Prepare the architecture for future MCP-style tools.

### Non-Goals

- Do not build a complete agent framework inside the tool layer.
- Do not implement a local arbitrary code sandbox as part of this work.
- Do not locally execute OpenAI built-in tools.
- Do not support every JSON Schema feature that a schema provider can emit.
- Do not require every future tool to be Pydantic-backed.
- Do not make every local tool concept OpenAI-specific.
- Do not perform broad unrelated refactors of fetching, cleaning, RAG, or
  pipeline code.

### API Ergonomics

The framework should support three levels of ergonomics:

1. Explicit production registration:
   - caller provides name, description, schema provider, handler, and serializer
   - most predictable and safest
2. Pydantic convenience registration:
   - caller provides `input_model: type[BaseModel]`
   - framework builds a schema provider around it
3. Decorator registration:
   - future `@tool(...)` helper
   - useful for small internal tools and tests

The current API should remain valid during migration:

- `OpenAIFunctionTool(...)`
- `ToolRegistry([tool])`
- `ToolRegistry(function_tools=[...], builtin_tools=[...])`
- `registry.schemas`
- `registry.run(...)`
- `registry.run_sync(...)`

Preferred future API:

- `FunctionTool(...)`
- `BuiltinTool.web_search(...)`
- `ToolRegistry(...).openai_tools`
- `OpenAIResponsesToolAdapter(registry).tools`
- `ToolExecutor(registry).run(...)`
- `ToolExecutor(registry).run_sync(...)`

### Performance Expectations

- Schema normalization must happen at registration time or first schema access,
  not on every model request.
- Normalized schemas must be cached.
- Tool lookup should be O(1) by function name.
- Validation cost is acceptable per tool call and should be delegated to the
  configured validator.
- Async execution should support bounded concurrency.
- Middleware overhead should be small and linear in the number of middleware
  objects.
- Avoid schema churn. Stable schemas help OpenAI cache strict schemas and reduce
  first-request latency.

### Thread Safety

- Tool definitions should be immutable after registration.
- `ToolRegistry` should be safe to share once constructed.
- `ToolExecutor` should hold concurrency primitives and per-executor policy.
- `ToolContext` should be per-call and never global.
- Handler state remains the responsibility of the handler owner unless wrapped
  by middleware.
- Schema caches should be protected by immutability, locks, or construction-time
  initialization.

### Async Behavior Guarantees

- Async handlers are awaited.
- Sync handlers in async execution should use an explicit policy:
  - default recommended policy: run sync handlers through `asyncio.to_thread`
  - stricter optional policy: reject sync handlers in async executor
- Sync execution must reject async-only handlers with a clear
  `ToolExecutionError`.
- Async cancellation must propagate to async handlers.
- Timeout errors should be distinct from ordinary handler errors.
- Middleware must either support async directly or be adapted through a clear
  sync-to-async wrapper.

## LLM Provider Extensibility

### Provider-Neutral Core

The following concepts should remain provider-neutral:

- `SchemaProvider`
- `FunctionTool`
- `ToolRegistry`
- `ToolExecutor`
- `ToolContext`
- `ToolResult`
- middleware
- errors
- validation
- timeout, retry, and concurrency policy
- local result serialization

These pieces should not assume OpenAI Responses message shapes, Anthropic tool
shapes, Gemini function declarations, or MCP descriptors.

### Provider-Specific Adapters

Provider-specific code should be isolated in adapter modules:

```text
core/tools/adapters/
  openai_responses.py
  openai_chat.py        # optional future adapter
  anthropic.py          # future adapter
  gemini.py             # future adapter
  mistral.py            # future adapter
  mcp.py                # future adapter
```

The first adapter should be `OpenAIResponsesToolAdapter`.

Responsibilities:

- compile provider-neutral function tools into OpenAI Responses tool schemas
- compile built-in OpenAI tools into Responses configs
- convert `ToolResult` into OpenAI `function_call_output`
- expose provider-specific warnings, such as strict schema constraints
- keep `.schemas` and `.openai_tools` compatibility during migration

Future adapters may compile the same `FunctionTool` differently. For example:

- Anthropic tools use their own tool schema envelope.
- Gemini function declarations use a different schema shape and naming.
- MCP tools may be remote descriptors rather than local Python handlers.
- OpenAI Chat Completions uses a nested `function` shape unlike Responses.

### Schema Compilation Targets

The architecture should distinguish between:

- source schema: emitted by `SchemaProvider`
- normalized internal schema: provider-neutral best-effort JSON Schema subset
- provider-compiled schema: adapted to a concrete LLM provider

Suggested naming:

- `SchemaProvider`: source schema and validation
- `SchemaNormalizer`: provider-neutral cleanup where possible
- `SchemaCompiler`: provider-specific compilation
- `OpenAIResponsesSchemaCompiler`: strict OpenAI Responses compiler

The current implementation plan can begin with one concrete compiler for OpenAI
Responses. Do not bake OpenAI strict requirements into the provider-neutral
objects themselves.

### Practical Compatibility Limits

Not every provider supports the same schema semantics.

Examples:

- OpenAI strict mode requires all object properties to be required and
  `additionalProperties: false`.
- Other providers may allow optional properties directly.
- Some providers support fewer JSON Schema keywords.
- Built-in tools are provider-specific and usually not portable.
- Tool call output message shapes differ between providers.

Therefore, portability should mean:

- local tool execution is reusable
- validation is reusable
- middleware is reusable
- provider adapters handle export differences

It should not mean every tool schema is guaranteed to work unchanged across
every provider.

## OpenAI Compatibility

### Responses API Function Tool Shape

Responses API function tools should be emitted as:

```json
{
  "type": "function",
  "name": "search_documents",
  "description": "Search local EU climate policy documents.",
  "parameters": {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": false
  },
  "strict": true
}
```

The current implementation already uses the Responses-style top-level shape.
It should add `strict: true` and normalize `parameters`.

### Strict Mode Requirements

For strict function calling:

- set `strict: true`
- every object schema must set `additionalProperties: false`
- every key in `properties` must be present in `required`
- optional fields must be represented as nullable fields
- schemas must use only the JSON Schema subset OpenAI supports

The framework should not rely on OpenAI implicitly normalizing schemas.
Normalization should happen locally so failures are caught in tests and at
startup.

### JSON Schema Limitations

The strict schema subset should allow:

- `string`
- `number`
- `integer`
- `boolean`
- `object`
- `array`
- `enum`
- `anyOf`
- supported string formats where needed
- supported numeric constraints where useful

The normalizer should reject or rewrite:

- `$ref` and `$defs` after normalization
- `allOf`
- `oneOf` unless safely convertible
- `not`
- `if` / `then` / `else`
- `dependentRequired`
- `dependentSchemas`
- `patternProperties`
- open-ended `additionalProperties`
- broad `dict[str, Any]`
- recursive schemas in the first implementation
- arbitrary custom JSON Schema extensions

### Chat Completions vs Responses API

Chat Completions commonly uses:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "...",
    "parameters": {}
  }
}
```

Responses API uses:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "...",
  "parameters": {},
  "strict": true
}
```

This framework should target Responses API only. If Chat Completions support is
ever needed, add a separate exporter rather than weakening the core model.

### Built-In Tools

Built-in tools should be represented separately from function tools.

Examples:

```json
{ "type": "web_search" }
```

```json
{
  "type": "web_search",
  "user_location": {
    "type": "approximate",
    "country": "IT",
    "city": "Turin"
  }
}
```

```json
{
  "type": "code_interpreter",
  "container": {
    "type": "auto",
    "memory_limit": "4g"
  }
}
```

Rules:

- built-ins are included in `openai_tools`
- built-ins are not included in the local executable function map
- `registry.get("web_search")` should return `None`
- `registry.is_builtin("web_search")` should return `True`
- built-in configs should be validated lightly by type-specific constructors
- raw dict passthrough should remain for compatibility

## Architecture

### Component Overview

```text
BaseTool
  |-- FunctionTool[InputT, ResultT]
  |-- BuiltinTool
  |-- future: RemoteMCPTool

SchemaProvider
  |-- PydanticSchemaProvider
  |-- RawJsonSchemaProvider
  |-- future: DataclassSchemaProvider
  |-- future: TypedDictSchemaProvider
  |-- future: MCPInputSchemaProvider

SchemaNormalizer
  |-- normalizes provider schemas into OpenAI strict schemas
  |-- validates the supported subset
  |-- caches normalized schemas

ToolRegistry
  |-- owns immutable tool definitions
  |-- exposes openai_tools / schemas
  |-- resolves executable function tools by name
  |-- tracks built-in tools by type

ToolExecutor
  |-- validates arguments
  |-- creates ToolContext
  |-- runs middleware
  |-- executes handlers
  |-- serializes ToolResult

ToolContext
  |-- per-call execution metadata
  |-- call_id, tool_name, raw arguments
  |-- request/user/session metadata
  |-- deadline, retry attempt, logger, trace context

ToolResult
  |-- success or error
  |-- serialized model-facing output
  |-- developer-facing value
  |-- duration and metadata
```

### BaseTool

`BaseTool` should define the shared interface:

- `name`
- `description`
- `kind`
- `to_openai_tool()`
- `metadata`

It should not define local execution. Local execution belongs only to
`FunctionTool`.

### FunctionTool

`FunctionTool` should represent a locally executable OpenAI function tool.

Responsibilities:

- hold name, description, schema provider, handler, result serializer
- expose normalized Responses API schema
- validate arguments through the schema provider
- expose whether handler is sync, async, or dual-compatible
- declare execution policy metadata:
  - timeout
  - retry policy
  - concurrency group
  - output mode
  - side-effect level
  - idempotency

It should not own the full execution pipeline. That belongs to `ToolExecutor`.

### BuiltinTool

`BuiltinTool` should represent an OpenAI-managed tool configuration.

Responsibilities:

- validate supported built-in `type`
- expose config through `to_openai_tool()`
- provide convenience constructors:
  - `BuiltinTool.web_search(...)`
  - `BuiltinTool.code_interpreter(...)`
  - `BuiltinTool.file_search(...)` later
  - `BuiltinTool.remote_mcp(...)` later

It should never have a local handler.

### SchemaProvider

Introduce a schema-provider abstraction to avoid over-restricting the framework
to Pydantic.

Required behavior:

- `json_schema() -> Mapping[str, object]`
- `validate(raw_args: Mapping[str, object]) -> InputT`
- `dump_validated(input_obj: InputT) -> Mapping[str, object]`

First implementation:

- `PydanticSchemaProvider`

Useful future implementations:

- `RawJsonSchemaProvider` with a custom validator
- `DataclassSchemaProvider`
- `TypedDictSchemaProvider`
- `MCPInputSchemaProvider`

Policy:

- Pydantic remains the recommended internal default.
- Core tool registration accepts any `SchemaProvider`.
- `OpenAIFunctionTool(input_model=...)` becomes a compatibility wrapper around
  `PydanticSchemaProvider`.

### ToolRegistry

Responsibilities:

- store all registered tools
- prevent name collisions among function tools
- prevent ambiguous collisions between function names and built-in types when
  they would confuse dispatch
- expose `openai_tools`
- keep `.schemas` as an alias for compatibility
- expose `get_function(name)`
- keep `get(name)` as a compatibility alias
- expose `is_builtin(type_or_name)`
- expose `function_names`
- expose `builtin_types`

The registry should not execute tools directly in the final architecture.
During migration, `run` and `run_sync` remain compatibility wrappers around a
default executor.

### ToolExecutor

Responsibilities:

- resolve function tools
- parse OpenAI arguments if needed
- validate arguments
- build `ToolContext`
- run middleware and hooks
- enforce timeouts and concurrency limits
- apply retry policy
- call the handler
- serialize the result
- wrap errors
- produce `ToolResult`
- produce Responses-compatible `function_call_output`

The agent loop should eventually delegate all tool-call handling to this class.

### ToolContext

Suggested fields:

- `tool_name: str`
- `call_id: str | None`
- `raw_arguments: Mapping[str, object]`
- `validated_arguments: object | None`
- `request_id: str | None`
- `user_id: str | None`
- `session_id: str | None`
- `metadata: Mapping[str, object]`
- `attempt: int`
- `deadline: float | None`
- `logger`
- `trace_id: str | None`

The context should be mutable only for framework-owned fields. Middleware may
attach metadata through a controlled dictionary.

### ToolResult

`ToolResult` should represent both successful and failed tool execution.

Suggested fields:

- `ok: bool`
- `tool_name: str`
- `call_id: str | None`
- `value: ResultT | None`
- `output: str`
- `error: ToolErrorPayload | None`
- `metadata: Mapping[str, object]`
- `duration_ms: float`

Output modes:

- `json`: serialize structured success and error payloads
- `text`: preserve plain text success output
- `raw_string`: pass through strings without wrapping
- `custom`: serializer decides

RAG should use text output for `search_documents` so the model receives the
existing context string.

### Middleware and Hooks

Middleware should be explicit and ordered.

Lifecycle:

1. `before_validate(context)`
2. `after_validate(context, validated)`
3. `before_call(context, validated)`
4. `after_call(context, result)`
5. `on_error(context, exception)`
6. `before_serialize(context, result)`
7. `after_serialize(context, tool_result)`

Repo-specific immediate uses:

- inject `directory` for `save_content_to_file`
- collect RAG sources from `SearchDocumentsResultModel`
- redact logs
- log fetch tool previews
- authorize risky tools
- collect metrics

## Schema System

### Normalization Pipeline

The normalization layer should:

1. Read schema from the configured `SchemaProvider`.
2. Deep-copy the schema.
3. Resolve `$ref` references using `$defs`.
4. Inline definitions where possible.
5. Remove `$defs`.
6. Convert nullable fields into OpenAI-compatible nullable schemas.
7. Convert safe literals into `enum` or supported `const` form.
8. Rewrite or reject unsupported unions.
9. Ensure every object has `additionalProperties: false`.
10. Ensure every object has `required` equal to all property keys.
11. Remove unsupported keywords.
12. Validate the result against the local OpenAI strict subset checker.
13. Cache the normalized schema.

### Defaults

Defaults are useful for local validation but should not be relied on for model
generation.

Recommended policy:

- Keep defaults in the schema provider for local validation.
- Remove `default` from OpenAI-facing schemas.
- Represent optional defaulted fields as required nullable fields if they can be
  missing from user intent.
- Document that strict OpenAI tools are not normal Python function signatures:
  every property is emitted by the model.

### Nullable Fields

Provider examples:

- Pydantic: `str | None = None`
- Dataclass: `str | None`
- Raw schema: `{"type": ["string", "null"]}`

OpenAI-facing strict schema should include the field in `required` and allow
`null`.

### Enums and Literals

Rules:

- Preserve small enums.
- Reject huge enums before request time.
- Convert `Literal["x"]` to `enum: ["x"]` unless target compatibility strongly
  prefers `const`.
- Keep enum descriptions on the parent field where possible.

### Unions

Rules:

- Allow simple nullable unions.
- Allow narrow `anyOf` unions when every branch is supported.
- Reject broad unions that make model behavior ambiguous.
- Prefer discriminated object unions only after tests prove OpenAI accepts the
  emitted structure reliably.

### Objects and Dicts

Rules:

- Closed objects are supported.
- Open dictionaries are not strict-compatible by default.
- `dict[str, str]` may be supported only if converted into an array of key/value
  objects or if strict mode is disabled for that tool.
- `dict[str, Any]` should raise `SchemaGenerationError` in strict mode.

### Caching

Cache key should include:

- provider class
- provider identity or schema fingerprint
- strict mode setting
- normalizer version
- transformer configuration

Pydantic-specific cache key may include:

- model class identity
- model schema fingerprint
- `by_alias` setting

## Execution System

### Async Execution Pipeline

1. Receive tool call.
2. Parse `tool_call.arguments` JSON.
3. Resolve function tool by `tool_call.name`.
4. Build `ToolContext`.
5. Run `before_validate`.
6. Validate arguments.
7. Run `after_validate`.
8. Run `before_call`.
9. Acquire global and per-tool concurrency semaphores.
10. Execute handler under timeout.
11. Apply retry policy if configured.
12. Run `after_call`.
13. Serialize result.
14. Return `ToolResult`.
15. Convert to `function_call_output` at the agent boundary.

### Sync Execution Pipeline

Same lifecycle as async, with sync-only guarantees:

- sync handlers run directly
- async-only handlers raise `ToolExecutionError`
- timeout is best-effort unless execution uses worker threads or processes

### Timeout Support

Policy levels:

- executor default timeout
- per-tool timeout
- per-call override through `ToolContext`

Async implementation should use `asyncio.timeout`.
Sync timeout should be documented as limited unless using a worker strategy.

### Cancellation Handling

- Async cancellation should propagate.
- Middleware should receive an `on_error` or `on_cancel` signal.
- Cancellation should not be retried unless explicitly configured, which should
  almost never be done.

### Retries

Default: no retries.

Allow retries only when:

- tool declares `idempotent=True`
- exception type is retryable
- retry budget is configured

Never retry filesystem writes, browser clicks, or external side-effect tools by
default.

### Concurrency Limits

Executor should support:

- global max concurrent tool calls
- per-tool max concurrent calls
- per-group limits for shared resources

Repo-specific examples:

- browser fetch tools may need low concurrency
- local search can use higher concurrency
- cleaning tools that mutate shared records should likely be serial

### Exception Wrapping

Raw exceptions from handlers should be wrapped in `ToolExecutionError`.

Validation failures should be wrapped in `ToolValidationError`.

Unknown names should become `UnknownToolError`.

Schema failures should happen before execution and become
`SchemaGenerationError`.

### Tool Tracing and Logging

Every call should emit structured events:

- `tool.started`
- `tool.validation_failed`
- `tool.validated`
- `tool.completed`
- `tool.failed`
- `tool.cancelled`
- `tool.serialized`

Logs should include:

- tool name
- call id
- duration
- argument preview
- output preview
- exception type
- retry attempt

Logs must support redaction.

## Error Handling

### Exception Types

`ToolFrameworkError`

- base class for framework errors

`ToolValidationError`

- invalid JSON
- missing required argument
- wrong argument type
- enum violation
- extra argument

`ToolExecutionError`

- handler failure
- timeout
- cancellation if converted
- retry exhaustion
- sync/async mode mismatch

`UnknownToolError`

- model requested a function not registered locally

`SchemaGenerationError`

- provider schema cannot be normalized to OpenAI strict schema
- unsupported JSON Schema construct
- schema too large
- invalid tool name

### Raise vs Return

Raise exceptions:

- during registration
- during schema generation
- during direct developer calls when `error_mode="raise"`
- in tests that intentionally validate failure behavior

Return structured `ToolResult`:

- inside model tool-call loops
- when `error_mode="return"`
- when the model can recover from the failure

Current compatibility:

- registry `run` and `run_sync` should continue returning `{"error": ...}` for
  unknown tools until agents are migrated.

### Serialized Error Format

JSON output mode:

```json
{
  "ok": false,
  "error": {
    "type": "ToolValidationError",
    "message": "Invalid arguments for search_documents.",
    "details": {},
    "retryable": false
  }
}
```

Text output mode:

```text
ToolValidationError: Invalid arguments for search_documents.
```

Use text output mode only where the existing agent prompt expects plain text.

## Typing

### Type Variables

Use:

- `InputT`
- `ResultT`
- `ValidatedT`
- `JsonValue`
- `JsonObject`

### FunctionTool Typing

`FunctionTool` should be generic:

- `FunctionTool[InputT, ResultT]`

Preferred handler shape:

- validated input object plus context

Compatibility handler shape:

- keyword arguments unpacked from validated input

### ParamSpec

Use `ParamSpec` for decorator ergonomics only.

Do not make core execution depend on preserving arbitrary Python callable
signatures. Core execution should operate on validated input objects.

### Avoiding Any

Public API should prefer:

- `Mapping[str, object]`
- `MutableMapping[str, object]`
- `JsonObject`
- `Sequence[BaseTool]`
- `Callable[..., ResultT]` only in compatibility adapters

Internal edges that integrate with OpenAI SDK objects may still need limited
`Any`.

## Extensibility

### Middleware

Middleware should be first-class and composable.

Initial built-in middleware:

- logging
- redaction
- argument mutation
- source collection for RAG
- output directory injection for fetch save tool

Future middleware:

- authorization
- rate limiting
- metrics
- OpenTelemetry tracing
- audit logging

### Authorization Layer

Authorize by:

- tool name
- side-effect level
- user id
- session id
- environment
- argument values

Suggested side-effect levels:

- `read_only`
- `writes_memory`
- `writes_files`
- `network`
- `browser`
- `external_side_effect`
- `executes_code`

### Rate Limiting

Support:

- per-tool limits
- per-user limits
- per-session limits
- shared resource group limits

Rate-limit failures should be structured and model-visible in agent loops.

### Observability

Metrics:

- call count
- validation failures
- execution failures
- timeout count
- retry count
- latency histogram
- concurrent in-flight count

Tracing:

- tool call span
- validation span
- handler span
- serialization span

### Pluggable Schema Transformers

`SchemaNormalizer` should accept a transformer chain.

Default chain:

- reference resolver
- unsupported keyword stripper/rejector
- nullable normalizer
- object strictifier
- required field normalizer
- final validator

Project-specific transformers can be added later.

### MCP-Style Future Compatibility

Future `RemoteMCPTool` should:

- export an OpenAI-compatible tool config
- not require a local Python handler
- use remote transport metadata
- support auth and resource scoping
- fit in `ToolRegistry` without changing `ToolExecutor`

The key architectural decision is that registry tools can be model-visible
without being locally executable.

## Testing Strategy

### Unit Tests

Add tests for:

- schema providers
- schema normalizer
- function tool schema export
- built-in tool schema export
- registry collision handling
- registry compatibility aliases
- executor sync dispatch
- executor async dispatch
- result serialization
- structured errors
- middleware ordering

### Integration Tests

Update or add tests for:

- RAG `search_documents` still works
- RAG source collection still works
- RAG `web_search` schema passthrough still works
- cleaning agent tool loop still works
- fetching agent async tool loop still works
- `save_content_to_file` still receives output directory injection
- unknown tools still return model-visible errors

### OpenAI Compatibility Tests

Golden snapshot tests:

- Responses API function tool shape
- `strict: true`
- recursive `additionalProperties: false`
- all object properties in `required`
- built-in `web_search`
- built-in `code_interpreter`
- no Chat Completions nested `function` object

Optional live smoke tests behind environment variables:

- one simple function tool call
- one `web_search` call if credentials and model support are available

### Property-Based Schema Tests

Use Hypothesis later for:

- idempotent normalization
- no `$defs` after normalization
- no unsupported object without `additionalProperties: false`
- required keys match property keys
- unsupported constructs raise clean errors

### Async Stress Tests

Test:

- many concurrent calls
- per-tool semaphore limits
- cancellation
- timeouts
- retry exhaustion
- sync handler in async executor policy

### Malformed Schema Tests

Test rejection of:

- broad dicts
- recursive models
- huge enums
- open objects
- unsupported `allOf`
- unsupported conditionals
- invalid tool names

## Packaging

### Recommended Directory Structure

```text
src/eu_climate_policy_rag/core/tools/
  __init__.py
  base.py
  function.py
  builtin.py
  registry.py
  executor.py
  schema.py
  providers.py
  context.py
  result.py
  errors.py
  middleware.py
  serialization.py
  decorators.py
```

Compatibility module:

```text
src/eu_climate_policy_rag/core/tooling.py
```

`tooling.py` should re-export compatibility names and contain minimal shim code.

### Module Boundaries

- `base.py`: shared tool protocols and base classes
- `function.py`: local executable tools
- `builtin.py`: OpenAI built-in tool configs
- `providers.py`: schema provider abstractions
- `schema.py`: normalization and compatibility checks
- `registry.py`: immutable tool catalog
- `executor.py`: execution pipeline
- `context.py`: per-call context
- `result.py`: result and output conversion
- `errors.py`: structured exceptions
- `middleware.py`: middleware protocols and built-ins
- `serialization.py`: result serializers
- `decorators.py`: developer convenience API

### Dependencies

Required:

- `pydantic>=2`
- existing OpenAI SDK dependency

Optional extras:

- `tools-test`: `hypothesis`, `pytest-asyncio`
- `tools-observability`: `opentelemetry-api`
- `tools-mcp`: future MCP dependency

## Developer Experience

### Decorators

Future decorator:

```text
@tool(name="search_documents", description="...")
```

Decorator should support:

- explicit schema provider
- explicit Pydantic model
- optional inferred model in dev mode
- output mode
- timeout
- side-effect level

In production, prefer explicit schemas over inference.

### Automatic Registration

Registry builder helpers can collect tools from:

- explicit sequences
- decorated functions
- class-backed toolboxes

Current builder modules should remain:

- `qa/tools.py`
- `collection/fetching/fetch_tools.py`
- `collection/cleaning/cleaning_tools.py`

They can migrate internally without changing callers.

### Schema Inspection

Add helpers:

- `registry.openai_tools`
- `registry.schemas`
- `registry.describe()`
- `registry.get_schema(name)`
- `registry.validate_openai_compatibility()`

### Debugging Helpers

Add:

- argument preview function with redaction
- result preview function
- schema pretty printer
- middleware trace log
- execution replay object for tests

### Development Warnings

Warn when:

- tool has empty description
- schema contains defaults removed from OpenAI-facing schema
- schema provider emits unsupported constructs
- sync handler is used in async executor through thread fallback
- built-in tools are present with incompatible parallel tool settings
- tool output is not JSON serializable in JSON mode

## Security

### Argument Validation

- Validate all function arguments before calling handlers.
- Reject unknown fields.
- Prefer strict local validation matching strict OpenAI schemas.
- Keep raw arguments for audit logs, with redaction.

### Dangerous Handler Isolation

Classify tools by side-effect level.

For risky tools:

- require authorization middleware
- add tighter timeout
- add lower concurrency
- log audit events
- avoid retries unless explicitly safe

### Execution Sandboxing

This framework should not pretend local Python handlers are sandboxed.

Options for future dangerous execution:

- OpenAI `code_interpreter`
- separate worker process
- containerized service
- restricted subprocess
- remote execution service

### Secret Handling

- Do not expose secrets in schemas or descriptions.
- Do not allow secrets as model-provided arguments unless absolutely necessary.
- Redact configured fields in logs.
- Pass credentials through server-side clients or context metadata.

### SSRF and Network Safety

Fetching/browser tools must validate URLs.

Block:

- localhost
- private IP ranges
- link-local IPs
- metadata service addresses
- unsupported URL schemes

Redirect targets must be revalidated.

Prefer OpenAI built-in `web_search` for general web lookup rather than exposing
raw network fetch tools to the model.

## Repo-Specific Migration Plan

### Phase 1: Add New Package Without Behavior Changes

Add `core/tools/` modules:

- errors
- result
- context
- providers
- schema normalizer
- base/function/builtin tool classes
- registry
- executor

Keep `core/tooling.py` working.

Acceptance:

- all existing tests pass
- no agent code changed yet

### Phase 2: Strict Schema Export

Change `OpenAIFunctionTool.schema` compatibility path to emit normalized strict
Responses schemas.

Acceptance:

- `strict: true` appears on function tools
- `additionalProperties: false` appears recursively
- existing tests updated for stricter schema expectations
- built-in tests still pass

### Phase 3: Registry Compatibility

Move registry internals to new `ToolRegistry`.

Keep:

- `.schemas`
- constructor compatibility
- `.get`
- `.run`
- `.run_sync`
- `.is_builtin`

Add:

- `.openai_tools`
- `.get_function`
- `.builtin_types`
- `.function_names`

Acceptance:

- existing consumers do not need changes

### Phase 4: Executor Integration

Introduce `ToolExecutor` and have registry compatibility methods delegate to it.

Acceptance:

- sync cleaning tests pass
- async fetching tests pass
- RAG tests pass
- unknown tool behavior remains compatible

### Phase 5: Agent Boundary Cleanup

Update `AbstractAgent` to use executor-generated Responses outputs.

Then update overrides:

- `ClimatePolicyAgent`
- `DocumentFetchAgent`

Replace custom behavior with middleware where practical.

Acceptance:

- source collection still works
- fetch logging still works
- save directory injection still works
- `function_call_output` shape unchanged

### Phase 6: Middleware and Observability

Add middleware for:

- logging
- redaction
- argument injection
- RAG source collection
- fetch result preview

Acceptance:

- less custom dispatch code in agents
- same external behavior

### Phase 7: Developer Experience

Add:

- decorators
- schema inspection helpers
- debug warnings
- documentation updates

Acceptance:

- existing builder modules can opt into new helpers gradually

## Risk Analysis

### Schema Compatibility Risk

Risk:

- Pydantic can emit schemas that OpenAI strict mode rejects.

Mitigation:

- local normalizer
- local strict subset validator
- golden tests
- fail fast at registration

### Optional Field Semantics Risk

Risk:

- strict mode requires all fields, which can surprise developers expecting
  normal optional Python parameters.

Mitigation:

- nullable-required normalization
- documentation
- development warnings
- careful tests around defaults and `None`

### Async Blocking Risk

Risk:

- sync handlers may block async agent loops.

Mitigation:

- explicit sync-in-async policy
- `asyncio.to_thread` default for compatibility
- warning for blocking mode

### Existing Behavior Regression Risk

Risk:

- RAG expects plain text output
- fetching expects JSON output
- cleaning expects Python objects in direct calls

Mitigation:

- output modes
- compatibility wrappers
- staged migration

### Built-In Tool Confusion Risk

Risk:

- built-ins are model-visible but not locally executable.

Mitigation:

- separate `BuiltinTool`
- separate registry maps
- clear `UnknownToolError` only for local function calls

### Over-Abstraction Risk

Risk:

- introducing providers, middleware, and executor may make simple tools harder
  to read.

Mitigation:

- keep compatibility constructors
- keep builder modules
- add explicit examples
- do not force decorators

## Recommended Final Architecture

The final architecture should be:

- `SchemaProvider` supplies schema and validation.
- `SchemaNormalizer` converts provider schemas into strict OpenAI-compatible
  Responses schemas.
- `FunctionTool` represents local executable function tools.
- `BuiltinTool` represents OpenAI-managed tools.
- `ToolRegistry` is an immutable model-visible catalog.
- `ToolExecutor` is the only execution pipeline.
- `ToolResult` is the only internal execution result type.
- Agents convert `ToolResult` into Responses `function_call_output`.
- Middleware handles cross-cutting behavior.

Pydantic should be the default provider, not the core abstraction.

## Future Roadmap

- MCP remote tool adapter
- OpenTelemetry tracing
- rate limiting middleware
- authorization middleware
- audit logging
- schema versioning
- schema diff reports
- tool call replay files
- parallel function-call execution in agent loops
- richer built-in tool constructors
- optional dataclass and TypedDict schema providers
