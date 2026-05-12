"""Unit tests for the OpenAI Responses tool adapter."""

import json

import pytest
from pydantic import BaseModel, Field

from eu_climate_policy_rag.core.tools import (
    BuiltinTool,
    FunctionTool,
    PydanticSchemaProvider,
    ToolRegistry,
    ToolResult,
)
from eu_climate_policy_rag.core.tools.adapters import (
    OpenAIResponsesSchemaCompiler,
    OpenAIResponsesToolAdapter,
)


class SearchInput(BaseModel):
    """Input model used by adapter tests."""

    query: str = Field(min_length=1)


def search_handler(query: str) -> dict[str, str]:
    """Return a small structured search result."""

    return {"query": query}


def build_registry() -> ToolRegistry:
    """Build a registry with one function tool and one built-in tool."""

    function_tool = FunctionTool(
        name="search_documents",
        description="Search documents",
        schema_provider=PydanticSchemaProvider(SearchInput),
        handler=search_handler,
    )
    return ToolRegistry(
        function_tools=[function_tool],
        builtin_tools=[BuiltinTool.web_search(user_location={"country": "IT"})],
    )


def test_openai_responses_adapter_exports_model_visible_tools() -> None:
    """Adapter should compile registry tools into Responses API tool configs."""

    adapter = OpenAIResponsesToolAdapter(build_registry())

    assert adapter.tools == [
        {
            "type": "function",
            "name": "search_documents",
            "description": "Search documents",
            "parameters": {
                "description": "Input model used by adapter tests.",
                "properties": {
                    "query": {"minLength": 1, "title": "Query", "type": "string"},
                },
                "required": ["query"],
                "title": "SearchInput",
                "type": "object",
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "web_search",
            "user_location": {"type": "approximate", "country": "IT"},
        },
    ]


def test_openai_responses_adapter_preserves_schema_aliases() -> None:
    """Compatibility aliases should expose the same Responses tool configs."""

    adapter = OpenAIResponsesToolAdapter(build_registry())

    assert adapter.openai_tools == adapter.tools
    assert adapter.schemas == adapter.tools


def test_openai_responses_adapter_does_not_depend_on_function_tool_export() -> None:
    """Adapter should own Responses schema compilation for function tools."""

    class FunctionToolWithoutOpenAIExport(FunctionTool[SearchInput, dict[str, str]]):
        def to_openai_tool(self) -> dict[str, object]:
            raise AssertionError("adapter should not call FunctionTool.to_openai_tool")

    function_tool = FunctionToolWithoutOpenAIExport(
        name="search_documents",
        description="Search documents",
        schema_provider=PydanticSchemaProvider(SearchInput),
        handler=search_handler,
    )
    adapter = OpenAIResponsesToolAdapter(
        ToolRegistry(function_tools=[function_tool]),
    )

    assert adapter.tools[0]["name"] == "search_documents"
    assert adapter.tools[0]["strict"] is True


def test_openai_responses_schema_compiler_compiles_function_tools() -> None:
    """Schema compiler should produce Responses function tool configs."""

    function_tool = build_registry().get_function("search_documents")
    assert function_tool is not None

    schema = OpenAIResponsesSchemaCompiler().compile_function_tool(function_tool)

    assert schema["type"] == "function"
    assert schema["name"] == "search_documents"
    assert schema["strict"] is True
    assert schema["parameters"]["additionalProperties"] is False


def test_openai_responses_adapter_converts_tool_results_to_function_call_output() -> None:
    """Adapter should convert ToolResult into Responses function_call_output."""

    result = ToolResult.success(
        tool_name="search_documents",
        value={"query": "climate"},
        call_id="call_123",
    )

    assert OpenAIResponsesToolAdapter.to_function_call_output(result) == {
        "type": "function_call_output",
        "call_id": "call_123",
        "output": json.dumps({"ok": True, "data": {"query": "climate"}}),
    }


def test_openai_responses_adapter_requires_call_id_for_function_call_output() -> None:
    """Responses function_call_output items require the model-provided call_id."""

    result = ToolResult.success(tool_name="search_documents", value={"query": "climate"})

    with pytest.raises(ValueError, match="call_id"):
        OpenAIResponsesToolAdapter.to_function_call_output(result)
