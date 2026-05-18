# Tool Framework

The project uses a provider-neutral tool framework under
`eu_climate_policy_rag.core.tools`.

Local Python tools are represented as `FunctionTool` objects. Provider-managed
tools, such as OpenAI web search, are represented as `BuiltinTool` objects.
Agents export both through provider adapters, currently
`OpenAIResponsesToolAdapter`.

## Core Concepts

- `FunctionTool`: a locally executable Python function exposed to the model
- `BuiltinTool`: a provider-executed tool that local Python does not dispatch
- `SchemaProvider`: validates raw model arguments and exposes JSON Schema
- `PydanticSchemaProvider`: schema provider backed by a Pydantic v2 model
- `ToolRegistry`: immutable catalog of local function tools and built-ins
- `ToolExecutor`: validates and runs local function tools
- `ToolResult`: structured success or failure result from one tool call
- `OpenAIResponsesToolAdapter`: exports registry tools to OpenAI Responses
  shapes and converts `ToolResult` values to `function_call_output` items

## Define A Function Tool

Use a Pydantic model for tool inputs when possible. The model gives the local
executor runtime validation and gives the provider adapter a JSON Schema source.

```python
from pydantic import BaseModel, Field

from eu_climate_policy_rag.core.tools import (
    FunctionTool,
    PydanticSchemaProvider,
)


class SearchInput(BaseModel):
    query: str = Field(description="Search query for policy documents.")
    num_results: int = Field(default=5, ge=1, le=20)


def search_documents(query: str, num_results: int = 5) -> dict[str, object]:
    return {
        "query": query,
        "num_results": num_results,
        "matches": [],
    }


search_tool = FunctionTool(
    name="search_documents",
    description="Search local EU climate policy documents.",
    schema_provider=PydanticSchemaProvider(SearchInput),
    handler=search_documents,
)
```

Tool names exported to OpenAI Responses must contain only letters, numbers,
underscores, and hyphens, and must be 1-64 characters long.

## Register Tools

Register local function tools and provider built-ins together in a
`ToolRegistry`.

```python
from eu_climate_policy_rag.core.tools import BuiltinTool, ToolRegistry


registry = ToolRegistry(
    function_tools=[search_tool],
    builtin_tools=[
        BuiltinTool.web_search(
            user_location={"city": "Turin", "country": "IT"},
        ),
    ],
)
```

Built-in tools are model-visible but provider-managed. They are exported to the
model request, but `ToolExecutor` does not run them locally.

Use `registry.describe()` when you need a serializable provider-neutral summary
for debugging, docs, or logs.

```python
summary = registry.describe()
print(summary["function_tools"])
print(summary["builtin_tools"])
```

## Export To OpenAI Responses

Use `OpenAIResponsesToolAdapter` at the provider boundary. The adapter compiles
function tools into strict OpenAI Responses schemas and passes built-in tool
configs through in Responses format.

```python
from eu_climate_policy_rag.core.tools.adapters import OpenAIResponsesToolAdapter


adapter = OpenAIResponsesToolAdapter(registry)
openai_tools = adapter.tools
```

`AbstractAgent` already does this internally. Most domain code should build a
native `ToolRegistry` and let the agent handle provider export.

## Execute Local Tools

Use `ToolExecutor` to validate arguments and run local function tools.

```python
from eu_climate_policy_rag.core.tools import ToolExecutor


executor = ToolExecutor(registry)
result = executor.run_sync(
    "search_documents",
    {"query": "EU climate neutrality", "num_results": 3},
    call_id="call_123",
)

assert result.ok
print(result.value)
```

The async executor path is `await executor.run(...)`. Sync execution rejects
async-only handlers with a structured tool execution error.

## Return Model-Visible Tool Outputs

Tool execution returns a `ToolResult`. To send a local tool result back to the
OpenAI Responses API, convert it through the adapter:

```python
function_call_output = OpenAIResponsesToolAdapter.to_function_call_output(result)
```

The result must include the model-provided `call_id`; Responses uses that ID to
match each output to the original function call.

## Error Handling

By default, `ToolExecutor` returns structured failure results instead of raising:

```python
missing = executor.run_sync("unknown_tool", {}, call_id="call_missing")
assert not missing.ok
print(missing.output)
```

Use `error_mode="raise"` when caller code should handle exceptions directly.

```python
executor.run_sync("unknown_tool", {}, error_mode="raise")
```

## Middleware

Middleware can observe or modify tool execution. Current project middleware is
used for fetch output-directory injection, RAG source collection, and cleaning
metrics.

Attach middleware to a registry when the behavior belongs to the tool catalog:

```python
registry = ToolRegistry(
    function_tools=[search_tool],
    middleware=[my_middleware],
)
```

Attach middleware to an executor for one execution boundary:

```python
executor = ToolExecutor(registry, middleware=[my_middleware])
```

Registry middleware is used when executor middleware is not supplied.

## Metrics

Use `ToolMetricsMiddleware` to record compact local tool execution metrics.

```python
from eu_climate_policy_rag.core.tools import ToolMetricsMiddleware


events = []
executor = ToolExecutor(
    registry,
    middleware=[ToolMetricsMiddleware(events.append)],
)
```

Each event includes the tool name, call ID, attempt number, success flag,
duration, and error type when execution fails.

For model response token usage and cost estimates, use
`ResponseUsageTracker` with explicit pricing values. OpenAI Responses expose
token usage on the response object, and pricing should stay configurable because
provider prices can change.

```python
from eu_climate_policy_rag.core.metrics import ModelPricing, ResponseUsageTracker


usage_tracker = ResponseUsageTracker(
    pricing_by_model={
        "gpt-4o-mini": ModelPricing(
            input_per_1m=0.15,
            cached_input_per_1m=0.075,
            output_per_1m=0.60,
        ),
    },
)
```

Pass `usage_tracker.record_response` as the Responses loop `on_response`
callback when constructing a loop directly.
